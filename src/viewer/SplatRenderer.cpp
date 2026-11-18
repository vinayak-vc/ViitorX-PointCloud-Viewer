#include "viewer/SplatRenderer.h"
#include "viewer/EmbeddedShaders.h"
#include "common/Log.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace pf {

using SteadyClock = std::chrono::steady_clock;
static double msSince(SteadyClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
}

// Order-preserving map float -> uint32 (flip sign bit for positives, invert all
// bits for negatives), so unsigned integer order matches float order.
static inline uint32_t floatToSortableKey(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

SplatRenderer::SplatRenderer() = default;

SplatRenderer::~SplatRenderer() {
    clear();
}

void SplatRenderer::startWorker() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = false;
    }
    worker_ = std::thread(&SplatRenderer::workerLoop, this);
}

void SplatRenderer::stopWorker() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
        jobPending_ = false;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void SplatRenderer::waitForWorkerIdle() {
    std::unique_lock<std::mutex> lk(mtx_);
    jobPending_ = false;                       // cancel anything not yet started
    cv_.wait(lk, [this] { return !busy_; });   // let a running job finish
    resultReady_ = false;
}

void SplatRenderer::clear() {
    stopWorker();
    if (quadVbo_) { glDeleteBuffers(1, &quadVbo_); quadVbo_ = 0; }
    if (quadVao_) { glDeleteVertexArrays(1, &quadVao_); quadVao_ = 0; }
    if (dataTex_) { glDeleteTextures(1, &dataTex_); dataTex_ = 0; }
    if (idxTex_)  { glDeleteTextures(1, &idxTex_);  idxTex_ = 0; }
    if (dataBuf_) { glDeleteBuffers(1, &dataBuf_); dataBuf_ = 0; }
    if (idxBuf_)  { glDeleteBuffers(1, &idxBuf_);  idxBuf_ = 0; }
    if (timerQ_[0]) { glDeleteQueries(2, timerQ_); timerQ_[0] = timerQ_[1] = 0; }
    timerInFlight_[0] = timerInFlight_[1] = false;
    positions_.clear();
    radii_.clear();
    mainIdx_.clear();
    fullOrder_.clear();
    workIdx_.clear();
    workerHasOrder_ = false;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        resultIdx_.clear();
        resultReady_ = false;
    }
    count_ = 0;
    haveLastVP_ = false;
    initialized_ = false;
}

bool SplatRenderer::init() {
    if (initialized_) return true;

    if (!shader_.loadFromSource(kSplatVertSrc, kSplatFragSrc)) {
        logError("SplatRenderer: Failed to compile splat shaders");
        return false;
    }

    // Static 2D quad geometry (-1,-1) to (1,1); the only vertex attribute.
    static const float kQuadCorners[8] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f
    };

    glGenVertexArrays(1, &quadVao_);
    glGenBuffers(1, &quadVbo_);
    glBindVertexArray(quadVao_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadCorners), kQuadCorners, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Splat data + draw list as texture buffers (GL 3.1+, fine on 3.3 core).
    glGenBuffers(1, &dataBuf_);
    glGenTextures(1, &dataTex_);
    glGenBuffers(1, &idxBuf_);
    glGenTextures(1, &idxTex_);
    glGenQueries(2, timerQ_);

    startWorker();
    initialized_ = true;
    logInfo("SplatRenderer: Initialized GPU texture-buffer splatting with background sort + culling");
    return true;
}

void SplatRenderer::upload(const SplatCloudData& data) {
    if (!initialized_ && !init()) return;
    const auto uploadT0 = SteadyClock::now();

    if (data.splats.empty()) {
        clear();
        return;
    }

    // Stop the sort thread from reading the old arrays before we replace them.
    waitForWorkerIdle();
    workerHasOrder_ = false;

    const size_t n = data.splats.size();
    positions_.resize(n);
    radii_.resize(n);

    // 4 texels (RGBA32F) per splat:
    //   t0 = (position.xyz, opacity)  t1 = (scale.xyz, 0)
    //   t2 = (rot w, x, y, z)         t3 = (colour.rgb, 0)
    // Positions are RELATIVE to the cloud centre for float precision; main.cpp
    // places the cloud with worldOffset = centre - sceneOrigin.
    std::vector<glm::vec4> texels(n * 4);
    for (size_t i = 0; i < n; ++i) {
        const auto& s = data.splats[i];
        glm::vec3 p((float)(s.position.x - data.center.x),
                    (float)(s.position.y - data.center.y),
                    (float)(s.position.z - data.center.z));
        positions_[i] = p;
        // 3 sigma bounding radius, used for frustum culling.
        radii_[i] = 3.0f * std::max((float)s.scale.x, std::max((float)s.scale.y, (float)s.scale.z));
        texels[i * 4 + 0] = glm::vec4(p, s.opacity);
        texels[i * 4 + 1] = glm::vec4(s.scale.x, s.scale.y, s.scale.z, 0.0f);
        texels[i * 4 + 2] = glm::vec4(s.quaternion[0], s.quaternion[1], s.quaternion[2], s.quaternion[3]);
        texels[i * 4 + 3] = glm::vec4(s.color[0], s.color[1], s.color[2], 0.0f);
    }

    glBindBuffer(GL_TEXTURE_BUFFER, dataBuf_);
    glBufferData(GL_TEXTURE_BUFFER, texels.size() * sizeof(glm::vec4), texels.data(), GL_STATIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, dataTex_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, dataBuf_);

    // Draw list starts as every splat in load order until the first job completes.
    mainIdx_.resize(n);
    for (size_t i = 0; i < n; ++i) mainIdx_[i] = static_cast<uint32_t>(i);
    glBindBuffer(GL_TEXTURE_BUFFER, idxBuf_);
    glBufferData(GL_TEXTURE_BUFFER, n * sizeof(uint32_t), mainIdx_.data(), GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_BUFFER, idxTex_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, idxBuf_);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    count_ = n;
    haveLastVP_ = false;   // force a job on the next draw

    timings_.uploadMs = msSince(uploadT0);
    logInfo("SplatRenderer: Uploaded " + std::to_string(count_) + " splats to GPU");
}

// Full back-to-front order for a view direction. Runs on the sort thread.
// Depth of splat p is dot(zAxis, p) + const, so only the view DIRECTION matters.
void SplatRenderer::sortFull(const glm::vec3& zAxis) {
    const size_t n = positions_.size();
    std::vector<uint32_t>& out = fullOrder_;
    out.resize(n);
    keys_.resize(n);
    keysTmp_.resize(n);
    idxTmp_.resize(n);

    // Ascending depth = farthest first: back-to-front for "over" compositing.
    for (size_t i = 0; i < n; ++i) {
        const glm::vec3& p = positions_[i];
        keys_[i] = floatToSortableKey(zAxis.x * p.x + zAxis.y * p.y + zAxis.z * p.z);
        out[i] = static_cast<uint32_t>(i);
    }

    // LSD radix sort, 4 passes x 8 bits. After an even number of passes the
    // result is back in `out`.
    for (int shift = 0; shift < 32; shift += 8) {
        size_t hist[256] = {0};
        for (size_t i = 0; i < n; ++i) hist[(keys_[i] >> shift) & 0xFFu]++;
        size_t sum = 0;
        for (int b = 0; b < 256; ++b) { size_t c = hist[b]; hist[b] = sum; sum += c; }
        for (size_t i = 0; i < n; ++i) {
            size_t dst = hist[(keys_[i] >> shift) & 0xFFu]++;
            keysTmp_[dst] = keys_[i];
            idxTmp_[dst] = out[i];
        }
        keys_.swap(keysTmp_);
        out.swap(idxTmp_);
    }

    workerDir_ = zAxis;
    workerHasOrder_ = true;
}

// Build the visible, back-to-front draw list. Runs on the sort thread.
void SplatRenderer::computeJob(const Job& job, std::vector<uint32_t>& out) {
    // Re-sort only when the view direction changed; otherwise reuse the order.
    if (!workerHasOrder_ || glm::dot(job.zAxis, workerDir_) <= 0.9999f) {
        sortFull(job.zAxis);
    }

    // Keep splats whose bounding sphere touches all six frustum planes AND whose
    // projected radius is at least minPx (screen-size LOD).
    out.clear();
    for (uint32_t i : fullOrder_) {
        const glm::vec3& p = positions_[i];
        const float r = radii_[i];

        bool inside = true;
        for (int k = 0; k < 6; ++k) {
            const glm::vec4& pl = job.planes[k];
            if (pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w < -r) { inside = false; break; }
        }
        if (!inside) continue;

        if (job.minPx > 0.0f) {
            // Projected radius in pixels. Ortho has no depth term.
            float px;
            if (job.ortho) {
                px = r * job.focalPx;
            } else {
                const float depth = -(job.depthRow.x * p.x + job.depthRow.y * p.y +
                                      job.depthRow.z * p.z + job.depthRow.w);
                px = (depth > 1e-4f) ? r * job.focalPx / depth : 1e30f;
            }
            if (px < job.minPx) continue;
        }
        out.push_back(i);
    }
}

void SplatRenderer::workerLoop() {
    std::unique_lock<std::mutex> lk(mtx_);
    while (true) {
        cv_.wait(lk, [this] { return stop_ || jobPending_; });
        if (stop_) return;

        jobPending_ = false;
        busy_ = true;
        const Job job = job_;
        lk.unlock();

        const auto t0 = SteadyClock::now();
        computeJob(job, workIdx_);
        const double ms = msSince(t0);

        lk.lock();
        std::swap(resultIdx_, workIdx_);   // publish the finished draw list
        resultReady_ = true;
        resultMs_ = ms;
        busy_ = false;
        cv_.notify_all();
    }
}

void SplatRenderer::draw(const glm::mat4& proj, const glm::mat4& view, const glm::vec2& viewportSize) {
    if (!initialized_ || count_ == 0) return;

    // 1) Pick up the newest finished draw list (never waits for the job).
    timings_.sortedThisFrame = false;
    bool haveNew = false;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (resultReady_) {
            std::swap(resultIdx_, mainIdx_);
            resultReady_ = false;
            haveNew = true;
            timings_.sortMs = resultMs_;
        }
    }
    if (haveNew && !mainIdx_.empty()) {
        // Only the visible splats' indices go up: 4 bytes each.
        glBindBuffer(GL_TEXTURE_BUFFER, idxBuf_);
        glBufferSubData(GL_TEXTURE_BUFFER, 0, mainIdx_.size() * sizeof(uint32_t), mainIdx_.data());
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
        timings_.sortedThisFrame = true;
    }
    timings_.visible = mainIdx_.size();
    if (mainIdx_.empty()) return;

    // 2) Submit a job when the camera moved and the worker is free. The
    //    matrix is cloud space (view includes the cloud's model offset).
    const glm::mat4 vp = proj * view;
    if (!haveLastVP_ || vp != lastVP_) {
        Job job;
        job.zAxis = glm::normalize(glm::vec3(view[0][2], view[1][2], view[2][2]));
        job.depthRow = glm::vec4(view[0][2], view[1][2], view[2][2], view[3][2]);
        job.focalPx = proj[1][1] * viewportSize.y * 0.5f;   // same focal as the shader
        job.ortho = proj[3][3] > 0.5f;
        job.minPx = minPixelRadius;

        // Gribb-Hartmann frustum planes from the rows of vp.
        const glm::vec4 row0(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
        const glm::vec4 row1(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
        const glm::vec4 row2(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
        const glm::vec4 row3(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);
        const glm::vec4 raw[6] = {
            row3 + row0, row3 - row0,   // left, right
            row3 + row1, row3 - row1,   // bottom, top
            row3 + row2, row3 - row2    // near, far
        };
        for (int k = 0; k < 6; ++k) {
            const float len = glm::length(glm::vec3(raw[k]));
            job.planes[k] = (len > 0.0f) ? raw[k] / len : raw[k];
        }

        std::lock_guard<std::mutex> lk(mtx_);
        if (!busy_ && !jobPending_) {
            job_ = job;
            jobPending_ = true;
            lastVP_ = vp;
            haveLastVP_ = true;
            cv_.notify_all();
        }
    }

    // 3) Draw the visible splats in the current order.
    shader_.use();
    shader_.setInt("uData", 0);   // texture unit 0: splat data
    shader_.setInt("uIdx", 1);    // texture unit 1: draw list
    shader_.setMat4("uProj", &proj[0][0]);
    shader_.setMat4("uView", &view[0][0]);
    shader_.setVec2("uViewportSize", viewportSize.x, viewportSize.y);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, dataTex_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, idxTex_);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE); // alpha-blended splats don't write depth

    glBindVertexArray(quadVao_);

    // GPU timer: collect any finished query (non-blocking), then time this draw
    // with a free slot. If both slots are still pending, skip timing this frame.
    for (int i = 0; i < 2; ++i) {
        if (!timerInFlight_[i]) continue;
        GLint available = 0;
        glGetQueryObjectiv(timerQ_[i], GL_QUERY_RESULT_AVAILABLE, &available);
        if (available) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(timerQ_[i], GL_QUERY_RESULT, &ns);
            timings_.gpuMs = static_cast<double>(ns) * 1e-6;
            timerInFlight_[i] = false;
        }
    }
    int slot = -1;
    if (!timerInFlight_[0]) slot = 0;
    else if (!timerInFlight_[1]) slot = 1;
    if (slot >= 0) glBeginQuery(GL_TIME_ELAPSED, timerQ_[slot]);

    // One instance per VISIBLE splat; the vertex shader maps gl_InstanceID
    // through the draw list to the splat's record.
    const auto drawT0 = SteadyClock::now();
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(mainIdx_.size()));
    timings_.drawCpuMs = msSince(drawT0);

    if (slot >= 0) {
        glEndQuery(GL_TIME_ELAPSED);
        timerInFlight_[slot] = true;
    }

    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

} // namespace pf
