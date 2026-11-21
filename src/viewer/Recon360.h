#pragma once
// 360 Reconstruction Studio — host side of the pf360 worker.
//
// PointForge does NOT train Gaussian Splats itself. A standalone Python worker
// (tools/pf360, copied beside the exe as pf360/) orchestrates FFmpeg, COLMAP
// and Nerfstudio Splatfacto as child processes and reports structured progress
// as JSON Lines on stdout (protocol v1, see docs/360-reconstruction.md). This
// module:
//   * locates the worker + a Python interpreter to run it,
//   * runs the short worker commands (doctor / probe / presets),
//   * writes the job manifest,
//   * launches `python -m pf360 run` inside a Windows Job Object (so cancel or
//     app exit can never leave orphaned COLMAP / training processes), parses
//     events and forwards them to the job queue.
//
// Viewer-only (never include from pfcore). Non-Windows builds get stubs.
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pf {

// User-configurable tool locations (Edit > Preferences > 360 Reconstruction).
// Empty = auto-detect (see tools/pf360/pf360/environment.py for the order).
struct Recon360Settings {
    std::string workerPython;   // interpreter that runs the pf360 worker (>= 3.8)
    std::string nsPython;       // python.exe of the Nerfstudio environment
    std::string colmapExe;
    std::string ffmpegExe;
    std::string ffprobeExe;
    std::string maskModelDir;   // optional masking add-on (scripts/pf360/setup_masking.ps1)
    std::string engineCatalogUrl;   // signed engine-pack catalog; "" = build default (EnginePacks.h)
};

// Folder that contains the `pf360` package (worker root), or "" if missing.
std::string recon360WorkerRoot();
// Interpreter used to run the worker ("" if none found).
std::string recon360FindPython(const Recon360Settings& s);

// ---- presets ------------------------------------------------------------------
struct Recon360Preset {
    std::string key, label, description, method, fallbackMethod, matchingMethod, costNote;
    double framesPerSecond = 3.0;
    int minFrames = 30, maxFrames = 300, imagesPerEquirect = 8, downscaleFactor = 2;
    int iterations = 30000, shDegree = 3;
    bool useScaleRegularization = false;
};
// Reads pf360/presets.json (single source of truth shared with the worker).
std::vector<Recon360Preset> recon360LoadPresets(std::string& err);
int recon360FramesForWindow(const Recon360Preset& p, double windowSeconds);

// Port of the Nerfstudio view selection (pf360/views.py, verified identical on
// 1800 crop combinations). Returns -1 when Nerfstudio would crash on the crop.
int recon360ViewsPerFrame(int imagesPerEquirect, float top, float bottom, float left, float right);
// The (yaw, pitch) view centres in degrees (yaw -180..180, pitch +down), same order
// as Nerfstudio. `unsupported` is set when Nerfstudio would crash on the crop.
std::vector<std::pair<float, float>> recon360ViewDirections(int imagesPerEquirect, float top, float bottom,
                                                            float left, float right, bool& unsupported);
int recon360PerspectiveSize(int width, int height, int imagesPerEquirect);

// ---- doctor -------------------------------------------------------------------
struct Recon360Check { std::string name, status, detail, fix; };
struct Recon360Doctor {
    bool ran = false;          // worker executed and returned JSON
    bool ready = false;        // every check PASS/WARN/SKIP
    bool cuda = false;
    bool masking = false;      // optional masking add-on installed + importable
    std::string maskModelDir, nsPython;   // resolved by the worker (for the mask preview)
    std::string error;         // why the doctor itself could not run
    std::vector<Recon360Check> checks;
    std::vector<std::string> methods;   // splatfacto variants the install registers
    std::string gpuName, nerfstudioVersion, torchVersion;
    int gpuTotalMB = 0, gpuFreeMB = 0;
};
// Spawns `python -m pf360 doctor --json` (slow: imports torch). Background thread only.
Recon360Doctor recon360RunDoctor(const Recon360Settings& s, bool quick);

// ---- probe --------------------------------------------------------------------
struct Recon360Probe {
    bool ok = false;           // readable + metadata parsed
    std::string error;
    std::string fileName, codec, pixelFormat, projection, projectionConfidence;
    int width = 0, height = 0;
    double fps = 0, durationSeconds = 0;
    long long frameCount = -1;
    uint64_t fileSizeBytes = 0;
    bool hasAudio = false, supported = false, requiresConfirmation = false;
    std::vector<std::string> reasons, warnings, errors;
};
// Spawns `python -m pf360 probe --json --quality`. Background thread only.
Recon360Probe recon360ProbeVideo(const Recon360Settings& s, const std::string& video, double start, double end);

// Decodes one downscaled RGB24 frame (width x width/2) for the wizard preview.
// Background thread only. Returns false + err on failure.
bool recon360ExtractFrame(const Recon360Settings& s, const std::string& video, double timeSeconds, int width,
                          std::vector<unsigned char>& rgb, int& outW, int& outH, std::string& err);

// ---- job ----------------------------------------------------------------------
struct Recon360JobSpec {
    std::string jobId, videoPath, outputDir, plyFileName;
    bool projectionConfirmed = false;
    double startSeconds = 0, endSeconds = -1;    // -1 = end of video
    int targetFrames = 90, imagesPerEquirect = 8;
    float cropTop = 0, cropBottom = 0, cropLeft = 0, cropRight = 0;
    std::string matchingMethod = "sequential";
    std::string qualityPreset = "balanced", method = "splatfacto", fallbackMethod;
    int iterations = 30000, downscaleFactor = 2, shDegree = 3, randomSeed = 42;
    bool useScaleRegularization = false;
    // Custom-only overrides (<0 = method default)
    float cullAlphaThresh = -1, densifyGradThresh = -1, cullScaleThresh = -1;
    std::string stopAfter;           // "" = full pipeline
    bool allowCache = true;
    // Automatic masking (optional add-on): pixels of these categories are ignored
    // by COLMAP and by Splatfacto's loss.
    bool maskEnabled = false;
    bool maskPeople = true, maskVehicles = false, maskAnimals = false, maskSky = false;
    int  maskDilatePx = 8;
};

// Writes masked-out pixels of one image tinted red into `overlayRgb` (same size as
// `rgb`, RGB24) using the masking add-on. Slow (loads the model): background
// thread only. `groups` is e.g. "people,vehicles".
bool recon360MaskPreview(const Recon360Settings& s, const std::vector<unsigned char>& rgb, int w, int h,
                         const std::string& groups, int dilatePx, std::vector<unsigned char>& overlayRgb,
                         float& maskedFraction, std::string& err);

// Live job state shared between the job worker thread (writer) and UI (reader).
struct Recon360Live {
    static constexpr int kStageCount = 8;
    static constexpr const char* kStages[kStageCount] = {"video_validation", "frame_extraction",
                                                         "panorama_projection", "masking", "camera_reconstruction",
                                                         "training", "export", "validation"};
    enum class StageState { Waiting, Running, Done, Skipped, Failed, Off };   // Off = optional stage disabled

    std::mutex mu;
    StageState stage[kStageCount] = {};
    int   currentStage = -1;
    float stagePercent = -1;       // -1 = indeterminate
    std::string stageMessage;
    std::vector<std::string> log;  // ring buffer of recent lines (events + stderr)
    std::vector<std::string> warnings;
    std::string outputPly, reportJson, workspace, finalMessage;
    uint64_t gaussianCount = 0, plyBytes = 0;
    double startedAt = 0;          // SDL ticks seconds at launch (UI clock)

    void addLog(const std::string& l) {
        log.push_back(l);
        if (log.size() > 600) log.erase(log.begin(), log.begin() + 100);
    }
};

const char* recon360StageLabel(int i);
// Overall fraction 0..1 for a stage index + its percent (weights by typical cost).
float recon360Overall(int stage, float pct);

// Writes the manifest to %LOCALAPPDATA%\ViitorX\PointForge\pf360-jobs\<jobId>.json
// (not into the output folder: the worker refuses non-empty folders it does
// not own, and copies the resolved manifest into the workspace itself).
bool recon360WriteManifest(const Recon360JobSpec& spec, const Recon360Settings& s, std::string& manifestPath,
                           std::string& err);

// Runs a job to completion. Blocking; call on the job worker thread. Honors
// `cancel` by asking the worker to stop ("cancel" on stdin), then terminating
// the whole Job Object after a grace period. `progress` receives overall 0..1
// (or <0 when indeterminate) + a status line.
bool recon360Run(const Recon360JobSpec& spec, const Recon360Settings& s, std::atomic<bool>& cancel,
                 Recon360Live& live, const std::function<void(float, const std::string&)>& progress,
                 std::string& err);

// ---- restore points: unfinished jobs, resumable after pause / crash / restart --------
struct Recon360Unfinished {
    Recon360JobSpec spec;            // rebuilt from the saved manifest
    std::string manifestPath;
    std::string lastStage;           // protocol id of the stage that was interrupted ("" = none started)
    std::string lastStatus;          // running (= crashed / app closed), cancelled (= paused), failed
    double updatedAt = 0;            // unix time of the last state change
};
// Jobs whose workspace exists and did not finish, newest first, one per workspace.
std::vector<Recon360Unfinished> recon360ListUnfinished();
// Rebuilds a job from a manifest written by recon360WriteManifest.
bool recon360SpecFromManifest(const std::string& manifestJson, Recon360JobSpec& spec, std::string& err);

// Call before the job queue shuts down at app exit: running jobs are then stopped
// immediately instead of waiting for a restore point to be saved (pausing waits).
void recon360SetShuttingDown();

// True when `plyPath` is the output of a pf360 job (its workspace report says
// so) — such files are Z-up (Nerfstudio orientation_method="up"), unlike the
// Y-down convention of most downloaded 3DGS files.
bool recon360IsZUpOutput(const std::string& plyPath);

// ---- batch helpers (GUI-free, tested by pf360hosttest) -------------------------
// MP4 / MOV / MKV files under `root` (UTF-8), sorted by path. Never descends into
// PointForge's own output (folders named PointForge360, or holding a 360 workspace).
// `truncated` is set when `maxCount` was hit.
std::vector<std::string> recon360FindVideos(const std::string& root, bool recursive, size_t maxCount, bool& truncated,
                                            const std::atomic<bool>* cancel = nullptr);
// What starting a job in `workspaceDir` producing `plyName` would do.
enum class Recon360WorkspaceState { New, Resume, Done, Foreign };
Recon360WorkspaceState recon360WorkspaceState(const std::string& workspaceDir, const std::string& plyName);

// Opens a folder / file in Explorer (no-op elsewhere).
void recon360Reveal(const std::string& path, bool selectFile);

} // namespace pf
