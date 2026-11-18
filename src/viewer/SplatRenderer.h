#pragma once
#include <GL/glew.h>
#include "io/SplatReader.h"
#include "viewer/Shader.h"
#include <glm/glm.hpp>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace pf {

// Gaussian splat renderer (OpenGL 3.3 core).
//
// Data layout:
//   * Splat records live on the GPU once per load, in a texture buffer (4 RGBA32F
//     texels per splat). They are never re-uploaded while the camera moves.
//   * The draw list is an index list (R32UI texture buffer) holding only the
//     VISIBLE splats in back-to-front order. Typically a few bytes per visible splat.
//
// A background thread builds the draw list. It re-sorts only when the view
// direction changes, and otherwise just filters the cached order against the
// view frustum. The render loop never waits for it: it uses the newest finished
// list. A list may lag the camera by a frame or two; splats are culled with a
// bounding-sphere test (3 sigma) so popping at the frustum edge is minimal.
class SplatRenderer {
public:
    SplatRenderer();
    ~SplatRenderer();

    // Initialize GL resources and compile shaders
    bool init();

    // Free GPU buffers, stop the sort thread and reset
    void clear();

    // Upload splat data to the GPU
    void upload(const SplatCloudData& data);

    // Render loaded splats
    void draw(const glm::mat4& proj, const glm::mat4& view, const glm::vec2& viewportSize);

    // LOD: splats whose projected radius is below this many pixels are not drawn.
    // Their contribution is sub-pixel, so skipping them trades little detail for
    // GPU time. 0 disables LOD.
    float minPixelRadius = 0.5f;

    size_t splatCount() const { return count_; }
    bool isReady() const { return initialized_ && count_ > 0; }

    // Per-frame timings for the Performance panel.
    struct Timings {
        double uploadMs = 0.0;          // CPU: last upload() (one-off per load)
        double sortMs = 0.0;            // CPU (sort thread): time of the last completed job
        bool   sortedThisFrame = false; // a new draw list was uploaded this frame
        size_t visible = 0;             // splats in the current draw list (after culling)
        double drawCpuMs = 0.0;         // CPU: time to submit the draw call
        double gpuMs = 0.0;             // GPU: draw time from a timer query (read 1-2 frames late)
    };
    const Timings& timings() const { return timings_; }

private:
    // One request to the sort thread: view direction + frustum planes (in cloud space).
    struct Job {
        glm::vec3 zAxis{0.0f};
        glm::vec4 planes[6]{};   // normalised (xyz = normal, w = d): dot(n,p)+d >= 0 is inside
        glm::vec4 depthRow{0.0f}; // view row 2: camera z = dot(xyz, p) + w (negative = in front)
        float focalPx = 0.0f;     // proj[1][1] * viewportH * 0.5 (pixels per unit at depth 1)
        bool  ortho = false;
        float minPx = 0.0f;       // LOD: skip splats whose projected radius is below this
    };

    // Sort thread
    void startWorker();
    void stopWorker();
    void workerLoop();
    void computeJob(const Job& job, std::vector<uint32_t>& out);
    void sortFull(const glm::vec3& zAxis);
    void waitForWorkerIdle();

    GLuint quadVao_ = 0;
    GLuint quadVbo_ = 0;

    // GPU data (texture buffers)
    GLuint dataBuf_ = 0, dataTex_ = 0;   // 4 x vec4 per splat
    GLuint idxBuf_ = 0, idxTex_ = 0;     // uint per visible splat (back-to-front)

    size_t count_ = 0;
    bool initialized_ = false;
    Shader shader_;

    // Per splat, read by the sort thread (read-only while a job runs; upload()
    // waits for idle before changing them).
    std::vector<glm::vec3> positions_;   // relative to cloud centre
    std::vector<float>     radii_;       // bounding radius = 3 * max(scale)

    // Main thread: submission state and the list currently drawn.
    glm::mat4 lastVP_{1.0f};
    bool      haveLastVP_ = false;
    std::vector<uint32_t> mainIdx_;      // draw list currently on the GPU

    // Sort thread -> main hand-off, guarded by mtx_.
    std::vector<uint32_t> resultIdx_;
    bool   resultReady_ = false;
    double resultMs_ = 0.0;

    // Sort thread state (guarded by mtx_ for job hand-off; the rest worker-only).
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool jobPending_ = false;
    bool busy_ = false;
    bool stop_ = false;
    Job  job_{};
    std::vector<uint32_t> workIdx_;      // worker only
    std::vector<uint32_t> fullOrder_;    // worker only: all splats, back-to-front for workerDir_
    glm::vec3 workerDir_{0.0f};          // worker only
    bool workerHasOrder_ = false;        // worker only
    std::vector<uint32_t> keys_, keysTmp_, idxTmp_;  // worker scratch

    Timings timings_;
    // Two GPU timer queries used alternately so reading one never stalls.
    GLuint timerQ_[2] = {0, 0};
    bool   timerInFlight_[2] = {false, false};
};

} // namespace pf
