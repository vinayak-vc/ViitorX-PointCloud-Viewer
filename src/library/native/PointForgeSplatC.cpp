// ---------------------------------------------------------------------------
// PointForgeSplatC.cpp — implementation of the splat half of PointForgeNative.
//
// Ports the viewer's splat pipeline (src/viewer/SplatRenderer.cpp + the GLSL in
// EmbeddedShaders.h) to the CPU so an engine without the viewer's shaders can
// draw the same result:
//   * frustum cull (bounding sphere, 3 sigma)
//   * screen-size LOD (projected radius in pixels)
//   * back-to-front order: radix sort on view direction. The order is cached
//     while the direction is unchanged, and a sorted copy of the hot data is kept
//     so the per-frame pass reads memory sequentially.
//   * 2D covariance (EWA splatting, same Jacobian clamp and +0.3 low-pass as the
//     viewer's vertex shader), inverted to a conic for the fragment falloff.
// The 3D covariance is view-independent and is computed once at open.
// ---------------------------------------------------------------------------
#define PF_UNITY_EXPORTS
#include "PointForgeSplatC.h"

#include "io/SplatReader.h"
#include "io/PackageFormat.h"
#include "common/Log.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

bool endsWithVxpc(const char* s) {
    const size_t n = std::strlen(s);
    return n >= 5 && std::strcmp(s + n - 5, ".vxpc") == 0;
}

// Order-preserving float -> uint32 key (same mapping as the viewer's radix sort).
inline uint32_t floatToSortableKey(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Per-splat data the per-frame pass needs, packed into one record (56 bytes)
// so one splat is one cache line span instead of five random reads.
struct SplatHot {
    float p[3];       // centred position
    float radius;     // 3-sigma bounding radius = 3 * max(scale)
    float sig[6];     // 3D covariance: s00 s01 s02 s11 s12 s22 (view-independent)
    float col[4];     // r, g, b, opacity
};
static_assert(sizeof(SplatHot) == 56, "SplatHot must stay 56 bytes");

struct Planes { std::array<glm::vec4, 6> p; };

Planes extractPlanes(const glm::mat4& m) {
    auto row = [&](int r) { return glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]); };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    Planes f;
    f.p[0] = r3 + r0; f.p[1] = r3 - r0;   // left, right
    f.p[2] = r3 + r1; f.p[3] = r3 - r1;   // bottom, top
    f.p[4] = r3 + r2; f.p[5] = r3 - r2;   // near, far
    for (auto& pl : f.p) {
        const float len = glm::length(glm::vec3(pl));
        if (len > 0.0f) pl /= len;
    }
    return f;
}

} // namespace

struct PFSplatCloud {
    uint64_t count = 0;
    glm::dvec3 center{0.0};
    glm::vec3 bmin{0.0f}, bmax{0.0f};
    bool hasHigherSH = false;

    std::vector<SplatHot> hot;      // load order (source of truth)
    std::vector<SplatHot> sorted;   // back-to-front copy for orderDir (drawn each frame)

    glm::vec3 orderDir{0.0f};
    bool      orderValid = false;

    // Radix-sort scratch.
    std::vector<uint32_t> keys, keysTmp, order, orderTmp;

    int32_t lastVisible = 0;
};

namespace {

// Builds `sorted` = `hot` reordered back-to-front for zAxis. Depth of p is
// dot(zAxis, p) + c, so only the direction matters; ascending depth = farthest first.
void sortFull(PFSplatCloud& c, const glm::vec3& zAxis) {
    const size_t n = c.hot.size();
    c.keys.resize(n);
    c.keysTmp.resize(n);
    c.order.resize(n);
    c.orderTmp.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const SplatHot& h = c.hot[i];
        c.keys[i] = floatToSortableKey(zAxis.x * h.p[0] + zAxis.y * h.p[1] + zAxis.z * h.p[2]);
        c.order[i] = static_cast<uint32_t>(i);
    }
    // LSD radix sort, 4 x 8 bits. Even pass count leaves the result in c.order.
    for (int shift = 0; shift < 32; shift += 8) {
        size_t hist[256] = {0};
        for (size_t i = 0; i < n; ++i) hist[(c.keys[i] >> shift) & 0xFFu]++;
        size_t sum = 0;
        for (int b = 0; b < 256; ++b) { size_t k = hist[b]; hist[b] = sum; sum += k; }
        for (size_t i = 0; i < n; ++i) {
            const size_t dst = hist[(c.keys[i] >> shift) & 0xFFu]++;
            c.keysTmp[dst] = c.keys[i];
            c.orderTmp[dst] = c.order[i];
        }
        c.keys.swap(c.keysTmp);
        c.order.swap(c.orderTmp);
    }
    c.sorted.resize(n);
    for (size_t i = 0; i < n; ++i) c.sorted[i] = c.hot[c.order[i]];
    c.orderDir = zAxis;
    c.orderValid = true;
}

} // namespace

extern "C" {

PF_API PFSplatCloud* PF_Splat_Open(const char* pathUtf8) {
    if (!pathUtf8 || !*pathUtf8) return nullptr;

    pf::SplatCloudData data;
    if (endsWithVxpc(pathUtf8)) {
        pf::PackageReader r;
        if (!r.Open(pathUtf8) || !r.Contains("splat.bin")) {
            pf::logError(std::string("PF_Splat_Open: no splat.bin in package ") + pathUtf8);
            return nullptr;
        }
        const std::vector<uint8_t> buf = r.Read("splat.bin");
        data = pf::SplatReader::loadSplatBinaryFromMemory(buf.data(), buf.size());
    } else {
        data = pf::SplatReader::loadFromFile(pathUtf8);
    }
    if (!data.ok || data.splats.empty()) {
        pf::logError(std::string("PF_Splat_Open: could not read splats from ") + pathUtf8);
        return nullptr;
    }

    PFSplatCloud* c = new (std::nothrow) PFSplatCloud();
    if (!c) return nullptr;

    const size_t n = data.splats.size();
    c->count = n;
    c->center = glm::dvec3(data.center.x, data.center.y, data.center.z);
    c->hot.resize(n);

    glm::vec3 bmin(1e30f), bmax(-1e30f);
    for (size_t i = 0; i < n; ++i) {
        const pf::GaussianSplat& g = data.splats[i];
        // Subtract the centre in double, then narrow: keeps precision at large offsets.
        const glm::vec3 p((float)(g.position.x - data.center.x),
                          (float)(g.position.y - data.center.y),
                          (float)(g.position.z - data.center.z));
        const glm::vec3 sc((float)g.scale.x, (float)g.scale.y, (float)g.scale.z);

        // quat stored as glm::quat(w, x, y, z); normalise defensively.
        glm::quat qRot(g.quaternion[0], g.quaternion[1], g.quaternion[2], g.quaternion[3]);
        const float qn = glm::length(glm::vec4(qRot.w, qRot.x, qRot.y, qRot.z));
        qRot = (qn > 1e-6f) ? glm::normalize(qRot) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

        // Sigma = M M^T with M = R S (view-independent).
        const glm::mat3 mMat = glm::mat3_cast(qRot) * glm::mat3(sc.x, 0, 0, 0, sc.y, 0, 0, 0, sc.z);
        const glm::mat3 sigMat = mMat * glm::transpose(mMat);

        SplatHot& h = c->hot[i];
        h.p[0] = p.x; h.p[1] = p.y; h.p[2] = p.z;
        h.radius = 3.0f * std::max(sc.x, std::max(sc.y, sc.z));
        h.sig[0] = sigMat[0][0]; h.sig[1] = sigMat[1][0]; h.sig[2] = sigMat[2][0];
        h.sig[3] = sigMat[1][1]; h.sig[4] = sigMat[2][1]; h.sig[5] = sigMat[2][2];
        h.col[0] = g.color[0]; h.col[1] = g.color[1]; h.col[2] = g.color[2];
        h.col[3] = g.opacity;

        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
        if (!g.sh_rest.empty()) c->hasHigherSH = true;
    }
    c->bmin = bmin;
    c->bmax = bmax;

    pf::logInfo("PF_Splat_Open: " + std::to_string(n) + " splats from " + pathUtf8);
    return c;
}

PF_API void PF_Splat_Close(PFSplatCloud* cloud) {
    delete cloud;
}

PF_API int32_t PF_Splat_GetInfo(PFSplatCloud* cloud, PFSplatInfo* out) {
    if (!cloud || !out) return 0;
    std::memset(out, 0, sizeof(*out));
    out->count = cloud->count;
    out->center[0] = cloud->center.x;
    out->center[1] = cloud->center.y;
    out->center[2] = cloud->center.z;
    for (int i = 0; i < 3; ++i) {
        out->boundsMin[i] = cloud->bmin[i];
        out->boundsMax[i] = cloud->bmax[i];
    }
    out->hasHigherSH = cloud->hasHigherSH ? 1u : 0u;
    return 1;
}

PF_API int32_t PF_Splat_LastVisibleCount(PFSplatCloud* cloud) {
    return cloud ? cloud->lastVisible : 0;
}

PF_API int64_t PF_Splat_CopyGpuRecords(PFSplatCloud* cloud, PFSplatGpuRecord* out, int64_t capacity) {
    if (!cloud || !out || capacity <= 0) return 0;
    static_assert(sizeof(PFSplatGpuRecord) == sizeof(SplatHot), "GPU record must mirror SplatHot");
    const int64_t n = std::min<int64_t>(capacity, static_cast<int64_t>(cloud->hot.size()));
    std::memcpy(out, cloud->hot.data(), static_cast<size_t>(n) * sizeof(SplatHot));
    return n;
}

PF_API int32_t PF_Splat_Update(PFSplatCloud* cloud, const PFSplatCamera* cam,
                               PFSplatInstance* out, int32_t capacity) {
    if (!cloud || !cam || !out || capacity <= 0) return 0;
    if (cam->viewportWidth <= 0.0f || cam->viewportHeight <= 0.0f) return 0;

    PFSplatCloud& c = *cloud;
    const glm::mat4 view = glm::make_mat4(cam->view);
    const glm::mat4 proj = glm::make_mat4(cam->proj);
    const glm::mat4 vp = proj * view;
    const bool ortho = cam->isOrtho != 0;
    const float vpW = cam->viewportWidth;
    const float vpH = cam->viewportHeight;
    const float fx = proj[0][0] * vpW * 0.5f;   // pixels per unit at depth 1 (same as shader)
    const float fy = proj[1][1] * vpH * 0.5f;
    const float minPx = cam->minPixelRadius;
    const float maxViewport = std::max(vpW, vpH);
    const float limX = 1.3f / proj[0][0];       // Jacobian clamp (standard 3DGS trick)
    const float limY = 1.3f / proj[1][1];

    int32_t cap = capacity;
    if (cam->maxInstances > 0 && (int32_t)cam->maxInstances < cap) cap = (int32_t)cam->maxInstances;

    // Back-to-front order: re-sort only when the view direction changes.
    const glm::vec3 zAxis = glm::normalize(glm::vec3(view[0][2], view[1][2], view[2][2]));
    if (!c.orderValid || glm::dot(zAxis, c.orderDir) <= 0.9999f) {
        sortFull(c, zAxis);
    }

    const Planes frustum = extractPlanes(vp);

    // Walk NEAR to far so that, when more splats are visible than `cap`, the
    // nearest ones are kept and the farthest (smallest on screen) are dropped.
    // The output is reversed at the end to restore back-to-front draw order.
    // (Walking far-to-near and stopping at cap dropped the foreground instead.)
    int32_t written = 0;
    int32_t visible = 0;
    for (auto it = c.sorted.rbegin(); it != c.sorted.rend(); ++it) {
        const SplatHot& h = *it;
        if (written >= cap) break;
        const float r = h.radius;
        const float px0 = h.p[0], py0 = h.p[1], pz0 = h.p[2];

        // Frustum: bounding sphere against the six planes.
        bool inside = true;
        for (const auto& pl : frustum.p) {
            if (pl.x * px0 + pl.y * py0 + pl.z * pz0 + pl.w < -r) { inside = false; break; }
        }
        if (!inside) continue;

        const glm::vec4 pc = view * glm::vec4(px0, py0, pz0, 1.0f);   // camera space (looks down -Z)
        // Behind or too close to the eye: unbounded footprint (same cut as the viewer).
        if (pc.z >= -0.2f) continue;
        const float depth = -pc.z;

        // Screen-size LOD: projected 3-sigma radius in pixels.
        const float pxRad = ortho ? r * fy : r * fy / depth;
        if (minPx > 0.0f && pxRad < minPx) continue;
        ++visible;

        // Jacobian of the projection (ortho: no depth term); T = J * W, W = view 3x3.
        float T[2][3];
        if (ortho) {
            for (int k = 0; k < 3; ++k) {
                T[0][k] = fx * view[k][0];
                T[1][k] = fy * view[k][1];
            }
        } else {
            const float invZ = 1.0f / pc.z;
            const float invZ2 = invZ * invZ;
            const float tx = std::clamp(pc.x * invZ, -limX, limX) * pc.z;
            const float ty = std::clamp(pc.y * invZ, -limY, limY) * pc.z;
            const float j00 = fx * invZ, j02 = -fx * tx * invZ2;   // J row 0
            const float j11 = fy * invZ, j12 = -fy * ty * invZ2;   // J row 1
            for (int k = 0; k < 3; ++k) {
                // W[r][k] = view[k][r]; T_ik = sum_r J_ir * W[r][k]
                T[0][k] = j00 * view[k][0] + j02 * view[k][2];
                T[1][k] = j11 * view[k][1] + j12 * view[k][2];
            }
        }

        // cov2D = T * Sigma * T^T, Sigma symmetric (s00 s01 s02 s11 s12 s22).
        const float* s = h.sig;
        float st[2][3];
        for (int i = 0; i < 2; ++i) {
            st[i][0] = s[0] * T[i][0] + s[1] * T[i][1] + s[2] * T[i][2];
            st[i][1] = s[1] * T[i][0] + s[3] * T[i][1] + s[4] * T[i][2];
            st[i][2] = s[2] * T[i][0] + s[4] * T[i][1] + s[5] * T[i][2];
        }
        const float covA = (T[0][0] * st[0][0] + T[0][1] * st[0][1] + T[0][2] * st[0][2]) + 0.3f;
        const float covB =  T[1][0] * st[0][0] + T[1][1] * st[0][1] + T[1][2] * st[0][2];
        const float covC = (T[1][0] * st[1][0] + T[1][1] * st[1][1] + T[1][2] * st[1][2]) + 0.3f;
        const float det = covA * covC - covB * covB;
        if (det <= 0.0f) continue;
        const float invDet = 1.0f / det;

        const float mid = 0.5f * (covA + covC);
        const float lambda = std::sqrt(std::max(0.1f, mid * mid - det));
        const float radiusPx = std::ceil(3.0f * std::sqrt(std::max(mid + lambda, mid - lambda)));
        // Drop giant footprints (camera inside / against the splat) instead of
        // clamping: a clamped quad cuts the Gaussian off at its edge, which shows
        // as hard-edged rectangles. Up to 2x the viewport the full quad is kept.
        if (radiusPx > 2.0f * maxViewport) continue;

        const glm::vec4 clip = proj * pc;
        if (clip.w == 0.0f) continue;

        PFSplatInstance& o = out[written++];
        o.position[0] = px0;
        o.position[1] = py0;
        o.position[2] = pz0;
        // World size of radiusPx at this depth: world-per-pixel = depth / fy (ortho: 1 / fy).
        o.worldRadius = ortho ? radiusPx / fy : radiusPx * depth / fy;
        o.ndc[0] = clip.x / clip.w;
        o.ndc[1] = clip.y / clip.w;
        o.depth = clip.z / clip.w;
        o.conic[0] = covC * invDet;
        o.conic[1] = -covB * invDet;
        o.conic[2] = covA * invDet;
        o.radiusPx = radiusPx;
        o.radiusNdc[0] = radiusPx * 2.0f / vpW;
        o.radiusNdc[1] = radiusPx * 2.0f / vpH;
        o.color[0] = h.col[0];
        o.color[1] = h.col[1];
        o.color[2] = h.col[2];
        o.color[3] = h.col[3];
    }

    // Collected near-to-far; callers blend in array order, so flip to back-to-front.
    std::reverse(out, out + written);

    c.lastVisible = visible;
    return written;
}

} // extern "C"
