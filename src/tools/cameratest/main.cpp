#include "viewer/Camera.h"
#include "viewer/CamPath.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

static bool nearVec(const glm::vec3& a, const glm::vec3& b, float eps = 1e-4f) {
    return glm::length(a - b) <= eps;
}

int main() {
    using pf::Camera;

    // 1. Horizon tilt offset (+5 deg) in keepLevel mode keeps front() and tilts right/up by 5 deg
    {
        Camera c;
        c.keepLevel = true;
        c.yaw = 0.0f;
        c.pitch = 0.0f;
        const glm::vec3 f0 = c.front();
        c.roll(5.0f);
        assert(std::fabs(c.tiltOffset - 5.0f) < 1e-5f);
        assert(nearVec(c.front(), f0, 1e-5f));
        // Rolling +5 deg about +Y tilts +Z toward +X (right-hand rule: +Y x +Z = +X)
        const float sin5 = std::sin(glm::radians(5.0f));
        const float cos5 = std::cos(glm::radians(5.0f));
        assert(nearVec(c.up(), glm::vec3(sin5, 0.0f, cos5), 1e-4f));
    }

    // 2. Drift-free after 10,000 random mouse moves with +5 deg tiltOffset
    {
        Camera c;
        c.keepLevel = true;
        c.tiltOffset = 5.0f;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-25.0f, 25.0f);
        for (int i = 0; i < 10000; ++i) {
            c.addYawPitch(dist(rng), dist(rng));
        }
        assert(std::fabs(c.tiltOffset - 5.0f) < 1e-5f);
        // Compare against a fresh camera at the exact same final (yaw, pitch, tiltOffset)
        Camera ref;
        ref.keepLevel = true;
        ref.yaw = c.yaw;
        ref.pitch = c.pitch;
        ref.tiltOffset = 5.0f;
        assert(nearVec(c.front(), ref.front(), 1e-5f));
        assert(nearVec(c.up(), ref.up(), 1e-5f));
        assert(nearVec(c.right(), ref.right(), 1e-5f));
    }

    // 3. Rotation locks block each respective axis
    {
        Camera c;
        c.keepLevel = true;
        c.yaw = 10.0f;
        c.pitch = 15.0f;
        c.tiltOffset = 5.0f;

        c.lockTilt = true;
        c.roll(12.0f);
        assert(std::fabs(c.tiltOffset - 5.0f) < 1e-5f);

        c.lockPitch = true;
        c.addYawPitch(20.0f, 30.0f);
        assert(std::fabs(c.pitch - 15.0f) < 1e-5f);
        assert(std::fabs(c.yaw - (10.0f + 20.0f * c.lookSpeed)) < 1e-4f);

        const float yAfter = c.yaw;
        c.lockYaw = true;
        c.addYawPitch(40.0f, 50.0f);
        assert(std::fabs(c.yaw - yAfter) < 1e-5f);
        assert(std::fabs(c.pitch - 15.0f) < 1e-5f);

        // Orbit also respects locks
        c.orbit(25.0f, 25.0f, glm::vec3(0.0f));
        assert(std::fabs(c.yaw - yAfter) < 1e-5f);
        assert(std::fabs(c.pitch - 15.0f) < 1e-5f);

        // levelHorizon resets tiltOffset to 0
        c.levelHorizon();
        assert(std::fabs(c.tiltOffset) < 1e-5f);
    }

    // 4. CamPath interpolates tilt
    {
        pf::CamPath path;
        pf::CamKey k0; k0.t = 0.0; k0.tilt = 0.0f;
        pf::CamKey k1; k1.t = 2.0; k1.tilt = 10.0f;
        path.keys.push_back(k0);
        path.keys.push_back(k1);
        pf::CamKey out;
        assert(path.sample(1.0, out));
        assert(std::fabs(out.tilt - 5.0f) < 1e-3f);
    }

    std::printf("cameratest: all 4 test groups PASSED\n");
    return 0;
}
