#include "viewer/Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>

namespace pf {

// Z-up world: yaw=0, pitch=0 looks +Y with +Z up and +X to the right.
// Roll-free orientation for a given yaw/pitch (forward = (sin y cos p, cos y cos p, sin p)).
glm::quat Camera::fromYawPitch(float yawDeg, float pitchDeg) {
    glm::quat qYaw   = glm::angleAxis(glm::radians(-yawDeg), glm::vec3(0, 0, 1));
    glm::quat qPitch = glm::angleAxis(glm::radians(pitchDeg), glm::vec3(1, 0, 0));
    return glm::normalize(qYaw * qPitch);
}

void Camera::sync() const {
    if (!orientValid_ || yaw != syncYaw_ || pitch != syncPitch_) {
        orient_ = fromYawPitch(yaw, pitch);
        syncYaw_ = yaw;
        syncPitch_ = pitch;
        orientValid_ = true;
    }
}

glm::vec3 Camera::front() const {
    sync();
    return glm::normalize(orient_ * glm::vec3(0, 1, 0));
}

glm::vec3 Camera::right() const {
    sync();
    return glm::normalize(orient_ * glm::vec3(1, 0, 0));
}

glm::vec3 Camera::up() const {
    sync();
    return glm::normalize(orient_ * glm::vec3(0, 0, 1));
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + front(), up());
}

glm::mat4 Camera::proj() const {
    if (isOrtho) {
        float right = orthoSize * aspect;
        return glm::ortho(-right, right, -orthoSize, orthoSize, nearZ, farZ);
    }
    return glm::perspective(glm::radians(fovY), aspect, nearZ, farZ);
}

void Camera::addYawPitch(float dYaw, float dPitch) {
    sync();
    const float yawRad   = glm::radians(dYaw * lookSpeed);
    const float pitchRad = glm::radians(dPitch * lookSpeed);

    // Local-frame rotations: post-multiply so each turn is about the camera's
    // current up (yaw) and right (pitch), not the world axes.
    //  - mouse right (dYaw > 0) turns right => negative rotation about local up
    //  - mouse up   (dPitch > 0) tilts up   => positive rotation about local right
    orient_ = glm::normalize(orient_
                             * glm::angleAxis(-yawRad,   glm::vec3(0, 0, 1))
                             * glm::angleAxis( pitchRad, glm::vec3(1, 0, 0)));

    // Refresh the serialised pose so saved bookmarks/paths match what is shown.
    glm::vec3 f = orient_ * glm::vec3(0, 1, 0);
    pitch = glm::degrees(std::asin(std::clamp(f.z, -1.0f, 1.0f)));
    yaw   = glm::degrees(std::atan2(f.x, f.y));
    syncYaw_ = yaw;
    syncPitch_ = pitch;
    orientValid_ = true;
}

void Camera::roll(float dDeg) {
    sync();
    // Local forward is +Y in camera space, so rolling is a local rotation about Y.
    // Forward and yaw/pitch are unchanged, so the yaw/pitch cache stays valid.
    orient_ = glm::normalize(orient_ * glm::angleAxis(glm::radians(dDeg), glm::vec3(0, 1, 0)));
}

void Camera::pan(float dxPx, float dyPx, float unitsPerPx) {
    position += (-right() * dxPx + up() * dyPx) * unitsPerPx;
}

void Camera::levelHorizon() {
    sync();
    // yaw/pitch describe the forward direction only, so rebuilding from them
    // gives the same forward with zero roll.
    orient_ = fromYawPitch(yaw, pitch);
}

void Camera::lookAt(const glm::vec3& target) {
    glm::vec3 dir = target - position;
    if (glm::length(dir) < 1e-6f) return;
    dir = glm::normalize(dir);
    // Inverse of the forward formula: front = (sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch))
    pitch = glm::degrees(std::asin(std::clamp(dir.z, -1.0f, 1.0f)));
    yaw   = glm::degrees(std::atan2(dir.x, dir.y));
    // sync() rebuilds a roll-free orientation from these on next use.
}

void Camera::orbit(float dYaw, float dPitch, const glm::vec3& pivot) {
    float dist = glm::length(position - pivot);
    if (dist < 1e-5f) dist = 1e-5f;
    addYawPitch(dYaw, dPitch);          // rotates about camera axes
    position = pivot - front() * dist;  // reposition so pivot stays centred
}

void Camera::screenRay(float mx, float my, int w, int h,
                       glm::vec3& outOrigin, glm::vec3& outDir) const {
    // Pixel -> normalised device coords (y flips: pixel 0 is top of window).
    float ndcX = (w > 0) ? (2.0f * mx / (float)w - 1.0f) : 0.0f;
    float ndcY = (h > 0) ? (1.0f - 2.0f * my / (float)h) : 0.0f;

    glm::mat4 inv = glm::inverse(viewProj());
    glm::vec4 nearH = inv * glm::vec4(ndcX, ndcY, -1.0f, 1.0f); // near plane
    glm::vec4 farH  = inv * glm::vec4(ndcX, ndcY,  1.0f, 1.0f); // far plane
    glm::vec3 nearP = glm::vec3(nearH) / nearH.w;
    glm::vec3 farP  = glm::vec3(farH)  / farH.w;

    outOrigin = nearP;
    outDir    = glm::normalize(farP - nearP);
}

} // namespace pf
