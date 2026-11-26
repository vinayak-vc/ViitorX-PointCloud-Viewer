#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace pf {

// First-person fly camera operating in "centred" space (world minus the octree
// cube centre) so coordinates stay small and float-friendly.
//
// Orientation is stored as a quaternion and mouse look rotates around the
// CAMERA's own axes (yaw about its up, pitch about its right), so turning is
// relative to where you are facing, not to the world.
//
// `yaw` / `pitch` are kept as the serialised pose (bookmarks, camera paths,
// saved views). They describe the forward vector. If external code assigns
// them, the orientation is rebuilt from them with roll cleared.
class Camera {
public:
    glm::vec3 position{0, 0, 0};
    float yaw   = -90.0f;   // degrees (pose of forward; see note above)
    float pitch = 0.0f;
    float fovY  = 60.0f;    // degrees
    float nearZ = 0.05f;
    float farZ  = 100000.0f;
    float aspect = 16.0f / 9.0f;
    bool  isOrtho = false;
    float orthoSize = 100.0f; // Half-height of ortho view
    float moveSpeed = 10.0f;     // units/sec
    float lookSpeed = 0.15f;     // degrees/pixel
    // true: mouse look yaws about WORLD up (+Z) and pitch is clamped to +-89 deg,
    // so the horizon never tilts (Unreal / game style). false: free 6-DOF look
    // about the camera's own axes, which slowly accumulates roll.
    bool  keepLevel  = true;
    float tiltOffset = 0.0f;   // degrees (-45..45), constant horizon tilt in level mode
    bool  lockTilt   = false;  // blocks roll (Z/X)
    bool  lockPitch  = false;  // blocks look up/down (pitch)
    bool  lockYaw    = false;  // blocks turn left/right (yaw)

    bool allLocked() const { return lockTilt && lockPitch && lockYaw; }
    void setAllLocked(bool v) { lockTilt = lockPitch = lockYaw = v; }

    glm::vec3 front() const;   // camera forward
    glm::vec3 right() const;   // camera right
    glm::vec3 up() const;      // camera up (roll follows the camera)
    glm::mat4 view() const;
    glm::mat4 proj() const;
    glm::mat4 viewProj() const { return proj() * view(); }

    // Rotate the camera by dYaw / dPitch (scaled by lookSpeed): about world up and
    // with no roll when keepLevel is on, else about its OWN axes.
    void addYawPitch(float dYaw, float dPitch);
    // Roll about the camera's FORWARD axis by dDeg (keeps forward, tilts the horizon).
    void roll(float dDeg);
    // Slide the camera sideways/up in its OWN view plane (middle-drag pan).
    // dxPx/dyPx are mouse deltas in pixels (dy > 0 = mouse moved down). The scene
    // follows the mouse, so the camera moves opposite to dx and with dy.
    void pan(float dxPx, float dyPx, float unitsPerPx);
    // Remove all roll: keep the current forward direction and make the horizon
    // level (world Z stays up, camera right stays horizontal).
    void levelHorizon();
    void moveLocal(const glm::vec3& delta) { position += delta; }

    // Point the camera at a target (sets yaw/pitch; position unchanged).
    void lookAt(const glm::vec3& target);
    // Orbit (turntable) around a pivot: rotate by dYaw/dPitch (scaled by
    // lookSpeed) while keeping the pivot centred and the distance to it fixed.
    void orbit(float dYaw, float dPitch, const glm::vec3& pivot);

    // Build a picking ray (in centred space) through pixel (mx, my) of a w×h
    // viewport. Works for both perspective and orthographic projection. Origin
    // lands on the near plane; dir is unit length and points into the scene.
    void screenRay(float mx, float my, int w, int h,
                   glm::vec3& outOrigin, glm::vec3& outDir) const;

private:
    // Rebuild orient_ from yaw/pitch/tiltOffset if they were changed externally.
    void sync() const;
    static glm::quat fromYawPitch(float yawDeg, float pitchDeg, float tiltDeg = 0.0f);

    mutable glm::quat orient_{1.0f, 0.0f, 0.0f, 0.0f};  // w, x, y, z
    mutable bool  orientValid_ = false;
    mutable float syncYaw_ = 0.0f;
    mutable float syncPitch_ = 0.0f;
    mutable float syncTilt_ = 0.0f;
};

} // namespace pf
