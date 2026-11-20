#include <GL/glew.h>   // must precede any system GL header
#include "viewer/Recon360Panel.h"
#include "common/FileDialog.h"
#include "common/Log.h"

#include "imgui.h"
#include <nlohmann/json.hpp>

#include <SDL.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace pf {

namespace {

const char* kMethods[] = {"splatfacto", "splatfacto-big"};
const char* kMatching[] = {"sequential", "exhaustive", "vocab_tree"};
const int kDownscales[] = {1, 2, 4, 8};
const ImVec4 kGood(0.45f, 0.85f, 0.45f, 1), kWarn(1.0f, 0.75f, 0.3f, 1), kBad(1.0f, 0.4f, 0.4f, 1),
             kDim(0.6f, 0.6f, 0.6f, 1);

double nowSec() { return (double)SDL_GetTicks64() / 1000.0; }

std::string fmtTime(double s) {
    if (s < 0) s = 0;
    int t = (int)s;
    char b[32];
    if (t >= 3600) snprintf(b, sizeof(b), "%d:%02d:%02d", t / 3600, (t / 60) % 60, t % 60);
    else snprintf(b, sizeof(b), "%02d:%02d", t / 60, t % 60);
    return b;
}

std::string fmtBytes(uint64_t b) {
    char buf[32];
    if (b >= (1ull << 30)) snprintf(buf, sizeof(buf), "%.2f GB", b / double(1ull << 30));
    else snprintf(buf, sizeof(buf), "%.1f MB", b / double(1ull << 20));
    return buf;
}

std::string fmtCount(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, ",");
    return s;
}

template <typename T> bool ready(const std::future<T>& f) {
    return f.valid() && f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

std::string stemSafe(const std::string& path) {
    std::string stem = fs::u8path(path).stem().u8string();
    std::string out;
    for (char c : stem) out += (std::isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '_';
    return out.empty() ? "scene" : out.substr(0, 80);
}

void helpMarker(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    ImGui::SetItemTooltip("%s", text);
}

void wrapped(const ImVec4& col, const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
}

} // namespace

bool Recon360Panel::isVideoFile(const std::string& path) {
    std::string ext = fs::u8path(path).extension().u8string();
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext == ".mp4" || ext == ".mov" || ext == ".mkv";
}

void Recon360Panel::show(const std::string& videoPath) {
    open = true;
    if (step >= 4 && job && !job->finished()) return;   // a run is in progress: just bring it up
    if (step >= 4) { step = 0; job.reset(); }
    if (!videoPath.empty()) {
        snprintf(videoBuf, sizeof(videoBuf), "%s", videoPath.c_str());
        probedPath.clear();     // force a new probe
        step = 0;
    }
}

void Recon360Panel::shutdownGL() {
    if (frameTex) { glDeleteTextures(1, &frameTex); frameTex = 0; }
}

void Recon360Panel::startProbe(Recon360PanelHost& host) {
    if (probeFut.valid()) return;
    probedPath = videoBuf;
    probe = Recon360Probe{};
    projectionConfirmed = false;
    frameRgb.clear();
    frameErr.clear();
    shutdownGL();          // drop the previous video's preview (UI thread owns the GL context)
    const Recon360Settings s = *host.settings;
    const std::string v = probedPath;
    probeFut = std::async(std::launch::async, [s, v]() { return recon360ProbeVideo(s, v, 0, -1); });
}

void Recon360Panel::startDoctor(Recon360PanelHost& host, bool quick) {
    if (doctorFut.valid()) return;
    const Recon360Settings s = *host.settings;
    doctorDeep = !quick;
    doctorFut = std::async(std::launch::async, [s, quick]() { return recon360RunDoctor(s, quick); });
}

void Recon360Panel::applyPreset(int idx) {
    presetIdx = idx;
    if (idx < 0 || idx >= (int)presets.size()) return;   // Custom keeps current values
    const Recon360Preset& p = presets[(size_t)idx];
    methodIdx = p.method == "splatfacto-big" ? 1 : 0;
    iterations = p.iterations;
    for (int i = 0; i < 4; ++i) if (kDownscales[i] == p.downscaleFactor) downscaleIdx = i;
    shDegree = p.shDegree;
    scaleReg = p.useScaleRegularization;
    viewsIdx = p.imagesPerEquirect == 14 ? 1 : 0;
    for (int i = 0; i < 3; ++i) if (p.matchingMethod == kMatching[i]) matchingIdx = i;
    useCullAlpha = useDensify = useCullScale = false;
    if (framesAuto && probe.ok)
        targetFrames = recon360FramesForWindow(p, (endSec > startSec ? endSec : probe.durationSeconds) - startSec);
}

void Recon360Panel::pollAsync() {
    if (ready(probeFut)) {
        probe = probeFut.get();
        if (probe.ok) {
            startSec = 0;
            endSec = (float)probe.durationSeconds;
            if (!projectBuf[0]) snprintf(projectBuf, sizeof(projectBuf), "%s", stemSafe(probedPath).c_str());
            if (!plyBuf[0]) snprintf(plyBuf, sizeof(plyBuf), "%s_splat.ply", stemSafe(probedPath).c_str());
            if (!parentBuf[0]) {
                const fs::path parent = fs::u8path(probedPath).parent_path() / "PointForge360";
                snprintf(parentBuf, sizeof(parentBuf), "%s", parent.u8string().c_str());
            }
            applyPreset(presetIdx < 3 ? presetIdx : 1);
        }
    }
    if (ready(doctorFut)) doctor = doctorFut.get();
    if (ready(maskFut)) {
        MaskResult r = maskFut.get();
        if (r.ok && r.rgb.size() == frameRgb.size()) {
            maskRgb = std::move(r.rgb);
            maskKey = r.key;
            maskFraction = r.masked;
            maskErr.clear();
            showMaskOverlay = true;
            frameDirty = true;
        } else {
            maskErr = r.ok ? "preview frame changed" : r.err;
        }
    }
    if (ready(frameFut)) {
        FrameResult r = frameFut.get();
        if (r.ok) {
            frameRgb = std::move(r.rgb);
            frameW = r.w;
            frameH = r.h;
            frameDirty = true;
            maskRgb.clear();          // overlay belongs to the previous frame
            maskKey.clear();
            showMaskOverlay = false;
        } else {
            frameErr = "Preview unavailable: " + r.err;
        }
    }
}

bool Recon360Panel::outputProblem(std::string& msg, bool& isResume) const {
    isResume = false;
    if (!projectBuf[0]) { msg = "Enter a project name."; return true; }
    for (const char* c = projectBuf; *c; ++c)
        if (std::strchr("<>:\"/\\|?*", *c)) { msg = "Project name contains a character Windows does not allow."; return true; }
    std::string ply = plyBuf;
    if (ply.size() < 5 || ply.substr(ply.size() - 4) != ".ply" || ply.find_first_of("<>:\"/\\|?*") != std::string::npos) {
        msg = "PLY file name must be a plain name ending in .ply.";
        return true;
    }
    std::error_code ec;
    const fs::path parent = fs::u8path(parentBuf);
    if (!parentBuf[0] || !fs::is_directory(parent, ec)) {
        // Allow a not-yet-existing parent if ITS parent exists (we create one level).
        if (!parentBuf[0] || !fs::is_directory(parent.parent_path(), ec)) {
            msg = "Output folder does not exist.";
            return true;
        }
    }
    const fs::path dir = parent / fs::u8path(projectBuf);
    if (fs::exists(dir, ec)) {
        if (fs::exists(dir / "state.json", ec)) {
            isResume = true;
            if (fs::exists(dir / "export" / fs::u8path(ply), ec)) {
                msg = "A reconstruction with this PLY name already exists in this project. Choose a new PLY name "
                      "or project to avoid overwriting it.";
                return true;
            }
            return false;
        }
        if (!fs::is_empty(dir, ec)) {
            msg = "That project folder already exists and is not a PointForge 360 workspace. Choose another name.";
            return true;
        }
    }
    return false;
}

void Recon360Panel::enqueue(Recon360PanelHost& host) {
    Recon360JobSpec sp;
    char id[64];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    snprintf(id, sizeof(id), "pf360_%04d%02d%02d_%02d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
    sp.jobId = id;
    sp.videoPath = probedPath;
    std::error_code ec;
    fs::create_directories(fs::u8path(parentBuf), ec);
    sp.outputDir = (fs::u8path(parentBuf) / fs::u8path(projectBuf)).u8string();
    sp.plyFileName = plyBuf;
    sp.projectionConfirmed = projectionConfirmed || !probe.requiresConfirmation;
    sp.startSeconds = startSec;
    sp.endSeconds = (endSec > 0 && endSec < probe.durationSeconds - 0.01) ? endSec : -1;
    sp.targetFrames = targetFrames;
    sp.imagesPerEquirect = viewsIdx == 1 ? 14 : 8;
    sp.cropTop = crop[0]; sp.cropBottom = crop[1]; sp.cropLeft = crop[2]; sp.cropRight = crop[3];
    sp.matchingMethod = kMatching[matchingIdx];
    sp.qualityPreset = presetIdx < (int)presets.size() ? presets[(size_t)presetIdx].key : "custom";
    sp.method = kMethods[methodIdx];
    if (presetIdx < (int)presets.size()) sp.fallbackMethod = presets[(size_t)presetIdx].fallbackMethod;
    sp.iterations = iterations;
    sp.downscaleFactor = kDownscales[downscaleIdx];
    sp.shDegree = shDegree;
    sp.randomSeed = seed;
    sp.useScaleRegularization = scaleReg;
    if (presetIdx == 3) {
        if (useCullAlpha) sp.cullAlphaThresh = cullAlpha;
        if (useDensify)   sp.densifyGradThresh = densifyGrad;
        if (useCullScale) sp.cullScaleThresh = cullScale;
    }
    sp.stopAfter = preprocessOnly ? "camera_reconstruction" : "";
    sp.maskEnabled = maskEnabled && doctor.masking && !maskGroups().empty();
    sp.maskPeople = maskPeople;
    sp.maskVehicles = maskVehicles;
    sp.maskAnimals = maskAnimals;
    sp.maskSky = maskSky;
    sp.maskDilatePx = maskDilate;
    job = host.jobs->enqueueRecon360(sp, *host.settings, autoLoad);
    jobStart = nowSec();
    jobEnd = 0;
    {
        std::lock_guard<std::mutex> lk(job->reconLive->mu);
        job->reconLive->startedAt = jobStart;
    }
    step = 4;
    logInfo("[pf360] queued " + sp.jobId + " -> " + sp.outputDir);
}

// ---------------------------------------------------------------------------
// preview texture
// ---------------------------------------------------------------------------
void Recon360Panel::uploadPreview() {
    if (!frameDirty || frameRgb.empty()) return;
    frameDirty = false;
    const bool overlay = showMaskOverlay && maskRgb.size() == frameRgb.size();
    if (!frameTex) glGenTextures(1, &frameTex);
    glBindTexture(GL_TEXTURE_2D, frameTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, frameW, frameH, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 overlay ? maskRgb.data() : frameRgb.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Recon360Panel::drawPreview(float width) {
    uploadPreview();
    const float h = width * 0.5f;
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (frameTex) {
        ImGui::Image((ImTextureID)(intptr_t)frameTex, ImVec2(width, h));
    } else {
        ImGui::Dummy(ImVec2(width, h));
        dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + h), IM_COL32(30, 30, 34, 255));
        const char* msg = frameFut.valid() ? "Loading preview frame..." :
                          (!frameErr.empty() ? frameErr.c_str() : "No preview");
        dl->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(170, 170, 170, 255), msg);
    }
    // Crop shading (top/bottom remove elevation rings; left/right narrow the yaw span).
    const ImU32 shade = IM_COL32(200, 40, 40, 90);
    if (crop[0] > 0) dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + h * crop[0]), shade);
    if (crop[1] > 0) dl->AddRectFilled(ImVec2(p0.x, p0.y + h * (1 - crop[1])), ImVec2(p0.x + width, p0.y + h), shade);
    if (crop[2] > 0) dl->AddRectFilled(ImVec2(p0.x + width * (1 - crop[2]), p0.y), ImVec2(p0.x + width, p0.y + h), shade);
    if (crop[3] > 0) dl->AddRectFilled(p0, ImVec2(p0.x + width * crop[3], p0.y + h), shade);
    // Perspective view centres.
    bool unsupported = false;
    auto dirs = recon360ViewDirections(viewsIdx == 1 ? 14 : 8, crop[0], crop[1], crop[2], crop[3], unsupported);
    for (size_t i = 0; i < dirs.size(); ++i) {
        const float x = p0.x + (dirs[i].first + 180.0f) / 360.0f * width;
        const float y = p0.y + (dirs[i].second + 90.0f) / 180.0f * h;
        dl->AddCircleFilled(ImVec2(x, y), 5.0f, IM_COL32(80, 200, 255, 230));
        char b[8];
        snprintf(b, sizeof(b), "%d", (int)i + 1);
        dl->AddText(ImVec2(x + 6, y - 7), IM_COL32(255, 255, 255, 230), b);
    }
    dl->AddRect(p0, ImVec2(p0.x + width, p0.y + h), IM_COL32(120, 120, 130, 255));
}

// ---------------------------------------------------------------------------
// environment settings (wizard + Preferences)
// ---------------------------------------------------------------------------
void Recon360Panel::drawEnvironmentSettings(Recon360PanelHost& host) {
    pollAsync();
    constexpr int kFields = 6;
    std::string* fields[kFields] = {&host.settings->workerPython, &host.settings->nsPython, &host.settings->colmapExe,
                                    &host.settings->ffmpegExe, &host.settings->ffprobeExe,
                                    &host.settings->maskModelDir};
    static const char* labels[kFields] = {"Worker Python", "Nerfstudio Python", "COLMAP", "FFmpeg", "FFprobe",
                                          "Masking model"};
    static const char* tips[kFields] = {
        "Interpreter that runs the pf360 worker (Python 3.8+, standard library only). Empty = py launcher / PATH.",
        "python.exe of the environment where Nerfstudio + CUDA PyTorch are installed. Empty = "
        "%LOCALAPPDATA%\\ViitorX\\PointForge\\engines\\pf360-env, then PATH.",
        "colmap.exe (3.11.x for Nerfstudio 1.1.5). Empty = PATH, then engines\\colmap-conda, engines\\colmap.",
        "ffmpeg.exe. Empty = PATH.", "ffprobe.exe. Empty = next to ffmpeg / PATH.",
        "Folder of the optional segmentation model (config.json + model.safetensors). Empty = "
        "%LOCALAPPDATA%\\ViitorX\\PointForge\\engines\\models\\upernet-convnext-base-ade20k, installed by "
        "scripts\\pf360\\setup_masking.ps1."};
    if (!pathBufsLoaded) {
        for (int i = 0; i < kFields; ++i) snprintf(pathBufs[i], sizeof(pathBufs[i]), "%s", fields[i]->c_str());
        pathBufsLoaded = true;
    }
    bool changed = false;
    for (int i = 0; i < kFields; ++i) {
        ImGui::PushID(i);
        ImGui::SetNextItemWidth(-90.0f * host.uiScale);
        if (ImGui::InputTextWithHint(labels[i], "auto-detect", pathBufs[i], sizeof(pathBufs[i]))) {
            *fields[i] = pathBufs[i];
            changed = true;
        }
        ImGui::SetItemTooltip("%s", tips[i]);
        if (i == 5) {               // a folder: type or paste it
            ImGui::PopID();
            continue;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("...")) {
            std::string f = openFileDialog("Executables\0*.exe\0All Files\0*.*\0");
            if (!f.empty()) {
                snprintf(pathBufs[i], sizeof(pathBufs[i]), "%s", f.c_str());
                *fields[i] = f;
                changed = true;
            }
        }
        ImGui::PopID();
    }
    if (changed && host.saveSettings) host.saveSettings();
    const std::string py = recon360FindPython(*host.settings);
    ImGui::TextDisabled("Worker: %s", recon360WorkerRoot().empty() ? "NOT FOUND" : recon360WorkerRoot().c_str());
    ImGui::TextDisabled("Python: %s", py.empty() ? "NOT FOUND" : py.c_str());
    if (doctorFut.valid()) {
        ImGui::TextDisabled(doctorDeep ? "Running full diagnostic (imports PyTorch, may take a minute)..."
                                       : "Running quick diagnostic...");
    } else {
        if (ImGui::Button("Quick check")) startDoctor(host, true);
        ImGui::SameLine();
        if (ImGui::Button("Full diagnostic")) startDoctor(host, false);
        ImGui::SetItemTooltip("Also imports PyTorch in the Nerfstudio env to verify CUDA and list training methods.");
    }
    if (!doctor.error.empty()) wrapped(kBad, "Diagnostic could not run: " + doctor.error);
    if (doctor.ran) {
        if (ImGui::BeginTable("##doctor", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            for (const auto& c : doctor.checks) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(c.name.c_str());
                ImGui::TableNextColumn();
                const ImVec4 col = c.status == "PASS" ? kGood : c.status == "FAIL" ? kBad : c.status == "WARN" ? kWarn : kDim;
                ImGui::TextColored(col, "%s", c.status.c_str());
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", c.detail.c_str());
                if (!c.fix.empty()) ImGui::SetItemTooltip("Fix: %s", c.fix.c_str());
            }
            ImGui::EndTable();
        }
        ImGui::TextColored(doctor.ready ? kGood : kBad, "Overall: %s", doctor.ready ? "READY" : "NOT READY");
        if (!doctor.ready)
            ImGui::TextWrapped("Hover a FAIL row for the fix. scripts\\pf360\\setup_environment.ps1 creates an "
                               "isolated Nerfstudio environment (asks before downloading ~10 GB).");
    }
}

// ---------------------------------------------------------------------------
// steps
// ---------------------------------------------------------------------------
void Recon360Panel::drawStepVideo(Recon360PanelHost& host) {
    ImGui::TextWrapped("Select a stitched equirectangular 360-degree video (MP4 / MOV / MKV). You can also drop a "
                       "video file onto the PointForge window.");
    ImGui::SetNextItemWidth(-110.0f * host.uiScale);
    ImGui::InputTextWithHint("##video", "path to 360 video", videoBuf, sizeof(videoBuf));
    ImGui::SameLine();
    if (ImGui::Button("Browse Video")) {
        std::string f = openFileDialog("360 Video\0*.mp4;*.mov;*.mkv\0All Files\0*.*\0");
        if (!f.empty()) snprintf(videoBuf, sizeof(videoBuf), "%s", f.c_str());
    }
    if (videoBuf[0] && probedPath != videoBuf && !probeFut.valid()) startProbe(host);

    if (probeFut.valid()) {
        ImGui::TextDisabled("Inspecting video (metadata, projection, sampled quality screen)...");
        return;
    }
    if (probedPath.empty()) return;
    if (!probe.ok) {
        wrapped(kBad, "Cannot use this file: " + (probe.error.empty() ? std::string("unknown error") : probe.error));
        return;
    }
    ImGui::Separator();
    const float avail = ImGui::GetContentRegionAvail().x;
    if (ImGui::BeginTable("##vinfo", 2, ImGuiTableFlags_SizingFixedFit)) {
        auto row = [](const char* k, const std::string& v) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextDisabled("%s", k);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(v.c_str());
        };
        char b[128];
        row("File", probe.fileName);
        snprintf(b, sizeof(b), "%d x %d", probe.width, probe.height); row("Resolution", b);
        row("Duration", fmtTime(probe.durationSeconds));
        snprintf(b, sizeof(b), "%.3f", probe.fps); row("FPS", probe.fps > 0 ? b : "unknown");
        row("Codec", probe.codec + " / " + probe.pixelFormat);
        row("Frames in file", probe.frameCount >= 0 ? fmtCount((uint64_t)probe.frameCount) : "unknown");
        row("File size", fmtBytes(probe.fileSizeBytes));
        row("Audio", probe.hasAudio ? "yes (ignored)" : "no");
        row("Projection", probe.projection + " (" + probe.projectionConfidence + ")");
        ImGui::EndTable();
    }
    for (const auto& r : probe.reasons) wrapped(probe.supported ? kDim : kBad, r);
    if (!probe.supported) {
        wrapped(kBad, "This input is not supported in this version: only stitched, monoscopic equirectangular video "
                      "can be reconstructed. Dual-fisheye / raw camera files must be stitched in the camera "
                      "vendor's software first.");
        return;
    }
    if (probe.requiresConfirmation)
        ImGui::Checkbox("I confirm this is a stitched equirectangular (2:1) 360 video", &projectionConfirmed);
    for (const auto& w : probe.warnings) wrapped(kWarn, "Warning: " + w);

    // representative frame
    if (frameRgb.empty() && !frameFut.valid() && frameErr.empty()) {
        const Recon360Settings s = *host.settings;
        const std::string v = probedPath;
        const double t = probe.durationSeconds * 0.5;
        frameFut = std::async(std::launch::async, [s, v, t]() {
            FrameResult r;
            r.ok = recon360ExtractFrame(s, v, t, 768, r.rgb, r.w, r.h, r.err);
            return r;
        });
    }
    drawPreview(std::min(avail, 640.0f * host.uiScale));

    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Capture guidance")) {
        ImGui::TextWrapped(
            "- Walk through the space: a camera that only rotates in place gives no depth (parallax).\n"
            "- Move steadily at walking pace; avoid fast turns and motion blur.\n"
            "- Keep the camera at a consistent height; avoid lighting changes.\n"
            "- Static scenes only: people or cars that move create ghosts and floaters (dynamic/4D "
            "reconstruction is not supported in this version).\n"
            "- The operator / tripod / selfie stick is usually at the bottom of the panorama: use the bottom "
            "crop on the next page. A crop cannot remove everything.");
    }
}

void Recon360Panel::drawStepProcessing(Recon360PanelHost& host) {
    const float dur = (float)probe.durationSeconds;
    ImGui::SeparatorText("Time range");
    ImGui::SetNextItemWidth(260.0f * host.uiScale);
    if (ImGui::DragFloatRange2("Start / end (s)", &startSec, &endSec, 0.1f, 0.0f, dur, "%.1f", "%.1f",
                               ImGuiSliderFlags_AlwaysClamp)) {
        if (endSec - startSec < 1.0f) endSec = std::min(dur, startSec + 1.0f);
        if (framesAuto && presetIdx < (int)presets.size())
            targetFrames = recon360FramesForWindow(presets[(size_t)presetIdx], endSec - startSec);
    }
    ImGui::SeparatorText("Frame sampling");
    ImGui::Checkbox("Follow quality preset", &framesAuto);
    helpMarker("Presets pick ~N frames per second of the selected range (Preview 1, Balanced/High 3), clamped. "
               "More frames are not always better: redundant frames add processing time without adding new "
               "camera positions.");
    ImGui::BeginDisabled(framesAuto);
    ImGui::SetNextItemWidth(200.0f * host.uiScale);
    ImGui::SliderInt("Target frames", &targetFrames, 2, 2000, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::EndDisabled();
    const double window = std::max(0.001f, endSec - startSec);
    ImGui::TextDisabled("= one frame every %.2f s (%.2f frames/s)", window / std::max(1, targetFrames),
                        targetFrames / window);
    if (probe.fps > 0 && targetFrames > window * probe.fps)
        wrapped(kWarn, "More frames requested than the range contains; all available frames will be used.");

    ImGui::SeparatorText("Perspective views");
    ImGui::RadioButton("8 views per frame", &viewsIdx, 0);
    ImGui::SameLine();
    ImGui::RadioButton("14 views per frame (more overlap)", &viewsIdx, 1);
    ImGui::TextDisabled("All views of one frame share the same camera position; Nerfstudio + COLMAP solve them as "
                        "a rig of rotated cameras.");
    ImGui::SeparatorText("Crop (removes view directions, Nerfstudio semantics)");
    ImGui::SetNextItemWidth(150.0f * host.uiScale);
    ImGui::SliderFloat("Top", &crop[0], 0.0f, 0.45f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f * host.uiScale);
    ImGui::SliderFloat("Bottom", &crop[1], 0.0f, 0.45f, "%.2f");
    helpMarker("Fraction of the vertical 180 degrees. Bottom crop is the usual way to drop the operator, tripod "
               "or selfie stick. Large crops remove whole rings of views.");
    ImGui::SetNextItemWidth(150.0f * host.uiScale);
    ImGui::SliderFloat("Left", &crop[2], 0.0f, 0.45f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f * host.uiScale);
    ImGui::SliderFloat("Right", &crop[3], 0.0f, 0.45f, "%.2f");
    helpMarker("Fraction of the horizontal 360 degrees: narrows the yaw span that views are generated in.");
    if (ImGui::SmallButton("Reset crop")) crop[0] = crop[1] = crop[2] = crop[3] = 0;

    const int ipe = viewsIdx == 1 ? 14 : 8;
    const int vpf = recon360ViewsPerFrame(ipe, crop[0], crop[1], crop[2], crop[3]);
    drawPreview(std::min(ImGui::GetContentRegionAvail().x, 560.0f * host.uiScale));
    if (vpf < 0) {
        wrapped(kBad, "This top + bottom crop combination removes a whole ring of views, which Nerfstudio's "
                      "projection cannot handle. Reduce the top or bottom crop.");
    } else if (vpf == 0) {
        wrapped(kBad, "The crop removes every view.");
    } else {
        ImGui::Text("%d frames x %d views = %s perspective images (%d x %d px each)", targetFrames, vpf,
                    fmtCount((uint64_t)targetFrames * (uint64_t)vpf).c_str(),
                    recon360PerspectiveSize(probe.width, probe.height, ipe),
                    recon360PerspectiveSize(probe.width, probe.height, ipe));
    }
    ImGui::SeparatorText("Feature matching (COLMAP)");
    ImGui::SetNextItemWidth(160.0f * host.uiScale);
    ImGui::Combo("Matching", &matchingIdx, kMatching, 3);
    helpMarker("sequential: matches neighbouring images in capture order (Nerfstudio's default for video, fast).\n"
               "exhaustive: every pair (robust, slow beyond ~500 images).\n"
               "vocab_tree: scalable retrieval (Nerfstudio downloads a vocabulary file on first use).");
    drawMasking(host);
}

std::string Recon360Panel::maskGroups() const {
    std::string g;
    auto add = [&](bool on, const char* name) { if (on) { if (!g.empty()) g += ","; g += name; } };
    add(maskPeople, "people");
    add(maskVehicles, "vehicles");
    add(maskAnimals, "animals");
    add(maskSky, "sky");
    return g;
}

void Recon360Panel::drawMasking(Recon360PanelHost& host) {
    ImGui::SeparatorText("Masking (optional)");
    if (!doctor.ran && !doctorFut.valid() && doctor.error.empty()) startDoctor(host, false);
    if (doctorFut.valid() && !doctor.ran) {
        ImGui::TextDisabled("Checking whether the masking add-on is installed...");
        return;
    }
    if (!doctor.masking) {
        wrapped(kDim, "Masking is not installed. It removes people, vehicles, animals or sky from the "
                      "reconstruction (moving objects otherwise leave ghosts and floaters). Install it with "
                      "scripts\\pf360\\setup_masking.ps1 (~0.5 GB: UperNet-ConvNeXt segmentation model, MIT "
                      "licence), then run the Full diagnostic again.");
        maskEnabled = false;
        return;
    }
    ImGui::Checkbox("Mask out moving / unwanted content", &maskEnabled);
    helpMarker("Every perspective view is segmented on the GPU (ADE20K classes). Masked pixels are ignored by "
               "COLMAP feature extraction and excluded from Gaussian training, so people walking past or passing "
               "traffic do not turn into ghosts.");
    ImGui::BeginDisabled(!maskEnabled);
    ImGui::Checkbox("People", &maskPeople);
    ImGui::SetItemTooltip("Includes what they ride (bicycles, motorbikes).");
    ImGui::SameLine();
    ImGui::Checkbox("Vehicles", &maskVehicles);
    ImGui::SetItemTooltip("Cars, buses, trucks, vans, bikes, boats. Also removes PARKED vehicles.");
    ImGui::SameLine();
    ImGui::Checkbox("Animals", &maskAnimals);
    ImGui::SameLine();
    ImGui::Checkbox("Sky", &maskSky);
    ImGui::SetItemTooltip("Sky is infinitely far and adds floaters. Masking it leaves the sky empty in the splat.");
    ImGui::SetNextItemWidth(160.0f * host.uiScale);
    ImGui::SliderInt("Grow mask (px)", &maskDilate, 0, 64, "%d", ImGuiSliderFlags_AlwaysClamp);
    helpMarker("Expands every masked region so edges (motion blur, shadows close to people) are excluded too.");
    const std::string groups = maskGroups();
    if (maskEnabled && groups.empty()) wrapped(kWarn, "Select at least one category, or turn masking off.");
    const std::string key = groups + "|" + std::to_string(maskDilate);
    if (maskFut.valid()) {
        ImGui::TextDisabled("Segmenting the preview frame (first run loads the model)...");
    } else {
        ImGui::BeginDisabled(frameRgb.empty() || groups.empty());
        if (ImGui::Button("Preview masks on this frame")) {
            const Recon360Settings s = *host.settings;
            const std::vector<unsigned char> rgb = frameRgb;
            const int w = frameW, h = frameH, dil = maskDilate;
            maskErr.clear();
            maskFut = std::async(std::launch::async, [s, rgb, w, h, groups, dil, key]() {
                MaskResult r;
                r.key = key;
                r.ok = recon360MaskPreview(s, rgb, w, h, groups, dil, r.rgb, r.masked, r.err);
                return r;
            });
        }
        ImGui::EndDisabled();
        if (!maskRgb.empty()) {
            ImGui::SameLine();
            if (ImGui::Checkbox("Show overlay", &showMaskOverlay)) frameDirty = true;
            if (maskKey == key)
                ImGui::TextDisabled("%.1f%% of the panorama is masked (red).", maskFraction * 100.0f);
            else
                ImGui::TextDisabled("Settings changed since the preview - preview again.");
        }
    }
    if (!maskErr.empty()) wrapped(kBad, "Mask preview failed: " + maskErr);
    ImGui::EndDisabled();
}

void Recon360Panel::drawStepQuality(Recon360PanelHost& host) {
    if (presets.empty() && presetErr.empty()) presets = recon360LoadPresets(presetErr);
    if (!presetErr.empty()) { wrapped(kBad, presetErr); return; }
    // GPU / environment summary (quick check runs automatically once).
    if (!doctor.ran && !doctorFut.valid() && doctor.error.empty()) startDoctor(host, false);
    if (doctorFut.valid()) ImGui::TextDisabled("Checking GPU and reconstruction environment...");
    else if (doctor.ran) {
        if (!doctor.gpuName.empty())
            ImGui::Text("GPU: %s  |  %d MB total, %d MB free", doctor.gpuName.c_str(), doctor.gpuTotalMB, doctor.gpuFreeMB);
        else
            wrapped(kWarn, "No NVIDIA GPU detected (nvidia-smi). Gaussian training needs a CUDA GPU.");
        ImGui::TextColored(doctor.cuda ? kGood : kBad, "CUDA in Nerfstudio env: %s", doctor.cuda ? "available" : "NOT available");
        if (!doctor.methods.empty()) {
            std::string m;
            for (const auto& s : doctor.methods) m += (m.empty() ? "" : ", ") + s;
            ImGui::TextDisabled("Installed training methods: %s", m.c_str());
        }
        if (!doctor.ready) {
            wrapped(kBad, "The reconstruction environment is not ready - see Environment below.");
        }
    } else if (!doctor.error.empty()) wrapped(kBad, doctor.error);

    ImGui::SeparatorText("Quality preset");
    const char* names[4] = {"Preview", "Balanced", "High Quality", "Custom"};
    for (int i = 0; i < 4; ++i) {
        if (i) ImGui::SameLine();
        if (ImGui::RadioButton(names[i], presetIdx == i)) applyPreset(i);
    }
    if (presetIdx < (int)presets.size()) {
        const Recon360Preset& p = presets[(size_t)presetIdx];
        ImGui::TextWrapped("%s", p.description.c_str());
        ImGui::BulletText("Model: %s%s", p.method.c_str(),
                          p.fallbackMethod.empty() ? "" : (" (falls back to " + p.fallbackMethod + " if not installed)").c_str());
        ImGui::BulletText("Training images: 1/%d resolution, %d iterations", p.downscaleFactor, p.iterations);
        ImGui::BulletText("Sampling: ~%.0f frames/s (%d-%d frames), %d views per frame", p.framesPerSecond,
                          p.minFrames, p.maxFrames, p.imagesPerEquirect);
        ImGui::BulletText("Cost: %s", p.costNote.c_str());
        ImGui::TextDisabled("No time estimate is shown: duration depends on GPU, image count and scene.");
    } else {
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::Combo("Model", &methodIdx, kMethods, 2);
        helpMarker("splatfacto: standard. splatfacto-big: lower cull-alpha (0.005) and densify (0.0005) thresholds "
                   "-> more Gaussians, more VRAM.");
        if (doctor.ran && !doctor.methods.empty() &&
            std::find(doctor.methods.begin(), doctor.methods.end(), kMethods[methodIdx]) == doctor.methods.end())
            wrapped(kBad, std::string(kMethods[methodIdx]) + " is not registered in the installed Nerfstudio.");
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::InputInt("Iterations", &iterations, 1000, 5000);
        iterations = std::clamp(iterations, 500, 200000);
        helpMarker("ns-train --max-num-iterations. Densification stops at step 15000 (Splatfacto default).");
        const char* ds[] = {"1 (full)", "2 (half)", "4 (quarter)", "8 (eighth)"};
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::Combo("Image downscale", &downscaleIdx, ds, 4);
        helpMarker("nerfstudio-data --downscale-factor: training resolution of the perspective images. Higher = less "
                   "VRAM, softer result.");
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::SliderInt("SH degree", &shDegree, 0, 3);
        helpMarker("--pipeline.model.sh-degree: view-dependent colour detail (0 = flat colour, 3 = full).");
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::InputInt("Random seed", &seed);
        seed = std::clamp(seed, 0, 2147483647);
        ImGui::Checkbox("Scale regularization", &scaleReg);
        helpMarker("--pipeline.model.use-scale-regularization: penalises needle-like Gaussians (max_gauss_ratio 10).");
        ImGui::Checkbox("##ca", &useCullAlpha); ImGui::SameLine();
        ImGui::BeginDisabled(!useCullAlpha);
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::InputFloat("Alpha pruning threshold", &cullAlpha, 0.001f, 0.01f, "%.4f");
        ImGui::EndDisabled();
        cullAlpha = std::clamp(cullAlpha, 0.0001f, 0.5f);
        helpMarker("--pipeline.model.cull-alpha-thresh: Gaussians below this opacity are removed (default 0.1, Big 0.005).");
        ImGui::Checkbox("##dg", &useDensify); ImGui::SameLine();
        ImGui::BeginDisabled(!useDensify);
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::InputFloat("Densify gradient threshold", &densifyGrad, 0.0001f, 0.001f, "%.5f");
        ImGui::EndDisabled();
        densifyGrad = std::clamp(densifyGrad, 0.00001f, 0.01f);
        helpMarker("--pipeline.model.densify-grad-thresh: lower = more splitting/cloning = more Gaussians "
                   "(default 0.0008, Big 0.0005).");
        ImGui::Checkbox("##cs", &useCullScale); ImGui::SameLine();
        ImGui::BeginDisabled(!useCullScale);
        ImGui::SetNextItemWidth(180.0f * host.uiScale);
        ImGui::InputFloat("Max Gaussian scale", &cullScale, 0.05f, 0.1f, "%.3f");
        ImGui::EndDisabled();
        cullScale = std::clamp(cullScale, 0.01f, 5.0f);
        helpMarker("--pipeline.model.cull-scale-thresh: Gaussians larger than this (scene units) are culled (default 0.5).");
        ImGui::TextDisabled("Not exposed: a maximum Gaussian budget - Nerfstudio 1.1.x Splatfacto has no such setting.");
    }
    ImGui::Spacing();
    ImGui::Checkbox("Stop after camera reconstruction (no training)", &preprocessOnly);
    helpMarker("Runs frame extraction, panorama projection and COLMAP only, so you can inspect the sparse "
               "point cloud and camera registration before spending GPU hours on training.");
    if (doctor.ran && doctor.gpuFreeMB > 0) {
        const int need = (int)((methodIdx == 1 ? 6000 : 4000) * (downscaleIdx == 0 ? 1.6 : downscaleIdx == 1 ? 1.0 : 0.6));
        if (doctor.gpuFreeMB < need)
            wrapped(kWarn, "Free VRAM (" + std::to_string(doctor.gpuFreeMB) + " MB) is below a rough guide of ~" +
                           std::to_string(need) + " MB for these settings. Reduce resolution, frames, use standard "
                           "Splatfacto, or close GPU-heavy applications. (VRAM alone does not guarantee a fit.)");
    }
    if (ImGui::CollapsingHeader("Environment")) drawEnvironmentSettings(host);
}

void Recon360Panel::drawStepOutput(Recon360PanelHost& host) {
    ImGui::SetNextItemWidth(260.0f * host.uiScale);
    ImGui::InputText("Project name", projectBuf, sizeof(projectBuf));
    ImGui::SetNextItemWidth(-110.0f * host.uiScale);
    ImGui::InputText("##parent", parentBuf, sizeof(parentBuf));
    ImGui::SameLine();
    if (ImGui::Button("Browse Folder")) {
        std::string d = openFolderDialog();
        if (!d.empty()) snprintf(parentBuf, sizeof(parentBuf), "%s", d.c_str());
    }
    ImGui::SetNextItemWidth(260.0f * host.uiScale);
    ImGui::InputText("Final PLY file name", plyBuf, sizeof(plyBuf));
    ImGui::Checkbox("Open in PointForge when finished", &autoLoad);
    ImGui::TextDisabled("Intermediate data (frames, COLMAP model, checkpoints) is always kept in the project folder "
                        "so retries can reuse finished stages. Delete the folder yourself when done.");
    const fs::path dir = fs::u8path(parentBuf) / fs::u8path(projectBuf);
    ImGui::Text("Workspace: %s", dir.u8string().c_str());
    std::string msg;
    bool resume = false;
    if (outputProblem(msg, resume)) wrapped(kBad, msg);
    else if (resume) wrapped(kWarn, "This folder is an existing 360 workspace: stages whose inputs and settings are "
                                    "unchanged will be reused (verified by content hashes).");
}

void Recon360Panel::drawStepRun(Recon360PanelHost& host) {
    if (!job) { step = 0; return; }
    const bool finished = job->finished();
    if (finished && jobEnd == 0) jobEnd = nowSec();
    Recon360Live& L = *job->reconLive;
    std::lock_guard<std::mutex> lk(L.mu);
    const int cur = L.currentStage;
    ImGui::Text("Current stage: %s", cur >= 0 ? recon360StageLabel(cur) : "Queued / starting worker");
    ImGui::Text("Elapsed: %s", fmtTime((finished ? jobEnd : nowSec()) - jobStart).c_str());
    const float w = ImGui::GetContentRegionAvail().x;
    if (!finished && L.stagePercent < 0)
        ImGui::ProgressBar(-1.0f * (float)ImGui::GetTime(), ImVec2(w, 0), "working (progress not reported)");
    else {
        char ov[64];
        snprintf(ov, sizeof(ov), "%.0f%% of stage", std::max(0.0f, L.stagePercent));
        ImGui::ProgressBar(finished ? 1.0f : L.stagePercent / 100.0f, ImVec2(w, 0), finished ? "" : ov);
    }
    if (!L.stageMessage.empty()) ImGui::TextDisabled("%s", L.stageMessage.c_str());
    ImGui::Spacing();
    if (ImGui::BeginTable("##stages", 2, ImGuiTableFlags_SizingFixedFit)) {
        for (int i = 0; i < Recon360Live::kStageCount; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(recon360StageLabel(i));
            ImGui::TableNextColumn();
            switch (L.stage[i]) {
                case Recon360Live::StageState::Waiting: ImGui::TextColored(kDim, "WAITING"); break;
                case Recon360Live::StageState::Running: ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "RUNNING"); break;
                case Recon360Live::StageState::Done:    ImGui::TextColored(kGood, "COMPLETED"); break;
                case Recon360Live::StageState::Skipped: ImGui::TextColored(kGood, "REUSED (cached)"); break;
                case Recon360Live::StageState::Off:     ImGui::TextColored(kDim, "OFF"); break;
                case Recon360Live::StageState::Failed:  ImGui::TextColored(kBad, "FAILED"); break;
            }
        }
        ImGui::EndTable();
    }
    if (doctor.ran && !doctor.gpuName.empty())
        ImGui::TextDisabled("GPU: %s (%d MB)", doctor.gpuName.c_str(), doctor.gpuTotalMB);
    for (const auto& wmsg : L.warnings) wrapped(kWarn, "Warning: " + wmsg);
    ImGui::SeparatorText("Log");
    if (ImGui::BeginChild("##r360log", ImVec2(0, 180.0f * host.uiScale), ImGuiChildFlags_Borders)) {
        for (const auto& l : L.log) ImGui::TextUnformatted(l.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4) ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    if (!finished) {
        if (ImGui::Button("Cancel")) JobQueue::cancel(job);
        ImGui::SetItemTooltip("Stops the worker and every COLMAP / training process it started. Intermediate data is kept.");
        ImGui::SameLine();
        if (ImGui::Button("Run in background")) {
            open = false;
            if (host.showJobsPanel) host.showJobsPanel();
        }
        if (job->state.load() == ConvertJob::State::Queued)
            ImGui::TextDisabled("Queued behind another job (one heavy job runs at a time).");
    } else {
        step = 5;
    }
}

void Recon360Panel::drawStepDone(Recon360PanelHost& host) {
    if (!job) { step = 0; return; }
    const ConvertJob::State st = job->state.load();
    Recon360Live& L = *job->reconLive;
    std::string ply, ws, report;
    uint64_t count = 0, bytes = 0;
    std::vector<std::string> warnings;
    {
        std::lock_guard<std::mutex> lk(L.mu);
        ply = L.outputPly; ws = L.workspace; report = L.reportJson; count = L.gaussianCount; bytes = L.plyBytes;
        warnings = L.warnings;
    }
    std::string method, iters, sparse;
    try {
        if (!report.empty()) {
            auto j = nlohmann::json::parse(report);
            method = j["training"].value("method", "");
            if (j["training"]["iterations"].is_number()) iters = std::to_string(j["training"]["iterations"].get<int>());
            if (j["processing"]["sparsePointCloud"].is_string()) sparse = j["processing"]["sparsePointCloud"].get<std::string>();
        }
    } catch (...) {}
    if (st == ConvertJob::State::Succeeded && !ply.empty()) {
        ImGui::TextColored(kGood, "Reconstruction complete");
        ImGui::Text("Output: %s", fs::u8path(ply).filename().u8string().c_str());
        ImGui::Text("Gaussians: %s", count ? fmtCount(count).c_str() : "unavailable");
        ImGui::Text("PLY size: %s", bytes ? fmtBytes(bytes).c_str() : "unavailable");
        ImGui::Text("Method: %s, %s iterations", method.empty() ? "unavailable" : method.c_str(),
                    iters.empty() ? "unavailable" : iters.c_str());
    } else if (st == ConvertJob::State::Succeeded) {
        ImGui::TextColored(kGood, "Stopped after camera reconstruction, as requested.");
    } else if (st == ConvertJob::State::Canceled) {
        ImGui::TextColored(kWarn, "Cancelled. Intermediate data and logs were kept.");
    } else {
        ImGui::TextColored(kBad, "Reconstruction failed");
        ImGui::TextWrapped("%s", job->message().c_str());
    }
    ImGui::Text("Total time: %s", fmtTime((jobEnd > 0 ? jobEnd : nowSec()) - jobStart).c_str());
    if (!ws.empty()) ImGui::TextDisabled("Workspace: %s", ws.c_str());
    for (const auto& w : warnings) wrapped(kWarn, "Warning: " + w);
    ImGui::Spacing();
    if (!ply.empty() && st == ConvertJob::State::Succeeded) {
        if (ImGui::Button("Open in PointForge") && host.openSplat) host.openSplat(ply);
        ImGui::SameLine();
    }
    if (!ws.empty()) {
        if (ImGui::Button("Open Output Folder")) recon360Reveal(ws, false);
        ImGui::SameLine();
        if (ImGui::Button("View Reconstruction Report"))
            recon360Reveal((fs::u8path(ws) / "reconstruction_report.txt").u8string(), false);
        ImGui::SameLine();
        if (ImGui::Button("Logs")) recon360Reveal((fs::u8path(ws) / "logs").u8string(), false);
    }
    if (!sparse.empty() && host.convertPointCloud) {
        if (ImGui::Button("Inspect sparse point cloud")) host.convertPointCloud(sparse);
        ImGui::SetItemTooltip("Opens COLMAP's sparse_pc.ply in the regular Convert dialog (point-cloud path).");
        ImGui::SameLine();
    }
    if (st == ConvertJob::State::Failed || st == ConvertJob::State::Canceled) {
        if (ImGui::Button("Retry")) {
            // Same settings + workspace: finished stages are reused via their hashes.
            job = host.jobs->enqueueRecon360(*job->recon, *host.settings, job->loadWhenDone);
            jobStart = nowSec();
            jobEnd = 0;
            step = 4;
            return;
        }
        ImGui::SameLine();
    }
    if (ImGui::Button("New reconstruction")) { step = 0; job.reset(); }
    ImGui::SameLine();
    if (ImGui::Button("Close")) open = false;
}

void Recon360Panel::draw(Recon360PanelHost& host) {
    pollAsync();
    if (!open) return;
    const float S = host.uiScale;
    ImGui::SetNextWindowSize(ImVec2(760 * S, 720 * S), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("360 Reconstruction Studio", &open, ImGuiWindowFlags_NoDocking)) {
        ImGui::End();
        return;
    }
    static const char* steps[6] = {"1 Video", "2 Processing", "3 Quality", "4 Output", "5 Run", "6 Done"};
    for (int i = 0; i < 6; ++i) {
        if (i) { ImGui::SameLine(); ImGui::TextDisabled(">"); ImGui::SameLine(); }
        if (i == step) ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "%s", steps[i]);
        else ImGui::TextDisabled("%s", steps[i]);
    }
    ImGui::Separator();
    if (presets.empty() && presetErr.empty()) presets = recon360LoadPresets(presetErr);

    const float footer = ImGui::GetFrameHeightWithSpacing() + 8 * S;
    if (ImGui::BeginChild("##r360body", ImVec2(0, -footer))) {
        switch (step) {
            case 0: drawStepVideo(host); break;
            case 1: drawStepProcessing(host); break;
            case 2: drawStepQuality(host); break;
            case 3: drawStepOutput(host); break;
            case 4: drawStepRun(host); break;
            default: drawStepDone(host); break;
        }
    }
    ImGui::EndChild();

    if (step <= 3) {
        ImGui::Separator();
        if (step > 0 && ImGui::Button("Back")) --step;
        ImGui::SameLine();
        bool canNext = false;
        std::string why;
        if (step == 0) {
            canNext = probe.ok && probe.supported && (!probe.requiresConfirmation || projectionConfirmed) &&
                      probedPath == videoBuf && !probeFut.valid();
        } else if (step == 1) {
            const int vpf = recon360ViewsPerFrame(viewsIdx == 1 ? 14 : 8, crop[0], crop[1], crop[2], crop[3]);
            canNext = vpf > 0 && targetFrames >= 2 && endSec > startSec;
        } else if (step == 2) {
            canNext = presetErr.empty() && !presets.empty();
            if (doctor.ran && !doctor.ready) why = "Environment not ready - the job will fail at the first missing tool.";
        } else if (step == 3) {
            bool resume = false;
            canNext = !outputProblem(why, resume);
        }
        ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - 190 * S);
        ImGui::BeginDisabled(!canNext);
        if (step < 3) {
            if (ImGui::Button("Next", ImVec2(90 * S, 0))) ++step;
        } else {
            if (ImGui::Button("Start Reconstruction", ImVec2(180 * S, 0))) enqueue(host);
        }
        ImGui::EndDisabled();
        if (!why.empty() && step == 2) ImGui::SetItemTooltip("%s", why.c_str());
    }
    ImGui::End();
}

} // namespace pf
