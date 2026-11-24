#pragma once
// "360° Reconstruction Studio" wizard window (Dear ImGui). Owns only UI state;
// all work runs in the shared JobQueue (Jobs.h) via pf360 (Recon360.h).
// main.cpp supplies a small host interface for app actions (load a splat,
// toasts, file dialogs live in pf::FileDialog).
#include "viewer/Jobs.h"
#include "viewer/Recon360.h"

#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

namespace pf {

struct Recon360PanelHost {
    JobQueue* jobs = nullptr;
    Recon360Settings* settings = nullptr;
    float uiScale = 1.0f;
    std::function<void(const std::string& ply)> openSplat;             // existing splat loading path
    std::function<void(const std::string& file)> convertPointCloud;    // existing Convert dialog (sparse PLY)
    std::function<void(const std::string& text, bool isError)> toast;
    std::function<void()> saveSettings;
    std::function<void()> showJobsPanel;
};

class Recon360Panel {
public:
    bool open = false;

    // Opens the wizard (optionally pre-filled with a dropped / chosen video).
    void show(const std::string& videoPath = "");
    // Draws the wizard window when open. Call once per frame inside the ImGui frame.
    void draw(Recon360PanelHost& host);
    // Tool-path settings + diagnostic (also embedded in Edit > Preferences).
    void drawEnvironmentSettings(Recon360PanelHost& host);
    // True when a dropped file looks like a video this wizard accepts.
    static bool isVideoFile(const std::string& path);
    // Releases the GL preview texture (call before GL context teardown).
    void shutdownGL();

private:
    void startProbe(Recon360PanelHost& host);
    void startDoctor(Recon360PanelHost& host, bool quick);
    void applyPreset(int idx);
    void pollAsync();
    bool outputProblem(std::string& msg, bool& isResume) const;
    void enqueue(Recon360PanelHost& host);
    void drawStepVideo(Recon360PanelHost& host);
    void drawStepProcessing(Recon360PanelHost& host);
    void drawStepQuality(Recon360PanelHost& host);
    void drawStepOutput(Recon360PanelHost& host);
    void drawStepRun(Recon360PanelHost& host);
    void drawStepDone(Recon360PanelHost& host);
    void drawPreview(float width);
    void uploadPreview();
    // Builds a job from the current wizard settings for one video (single + batch).
    Recon360JobSpec buildSpec(const std::string& video, const Recon360Probe& pr, const std::string& outputDir,
                              const std::string& plyName, bool confirmed, bool fullLength, int index) const;

    // ---- restore points: unfinished jobs (paused / app closed / crashed / failed)
    std::vector<Recon360Unfinished> unfinished;
    bool unfinishedDirty = true;
    void drawUnfinished(Recon360PanelHost& host);
    void resumeJobs(Recon360PanelHost& host, const std::vector<Recon360JobSpec>& specs);

    // ---- batch: every 360 video under a folder, same settings, queued in order
    bool batchMode = false;
    struct BatchItem {
        std::string path, rel;              // absolute / relative to the root (UTF-8)
        Recon360Probe probe;
        bool probed = false, selected = false;
        std::string project, plan;          // workspace name + what Start will do ("new", "resume", "skip: ...")
        bool willRun = false;
    };
    struct BatchScan {                      // shared with the scan/probe thread
        std::mutex mu;
        std::vector<BatchItem> items;
        bool listed = false, done = false;
        std::string error;
        std::atomic<bool> cancel{false};
    };
    std::shared_ptr<BatchScan> bscan;
    std::future<void> bscanFut;
    std::vector<BatchItem> batch;           // UI copy (selection edits live here)
    bool batchSynced = false;
    char batchRootBuf[1024] = {};
    bool batchRecursive = true;
    bool batchConfirm = false;              // "all selected videos are stitched equirectangular"
    std::vector<std::shared_ptr<ConvertJob>> batchJobs;
    std::vector<std::string> batchJobNames;   // parallel to batchJobs (relative video path)
    double batchStart = 0;
    void startBatchScan(const Recon360Settings& s);
    void batchToFirstVideo(Recon360PanelHost& host);
    void syncBatch();
    void planBatch();
    void enqueueBatch(Recon360PanelHost& host);
    void drawBatchVideo(Recon360PanelHost& host);
    void drawBatchOutput(Recon360PanelHost& host);
    void drawBatchRun(Recon360PanelHost& host);
    int batchSelectedCount() const;
    bool batchNeedsConfirm() const;

    int step = 0;   // 0 video, 1 processing, 2 quality, 3 output, 4 processing(run), 5 done

    // video
    char videoBuf[1024] = {};
    std::string probedPath;
    Recon360Probe probe;
    std::future<Recon360Probe> probeFut;
    bool projectionConfirmed = false;
    // preview frame
    struct FrameResult { bool ok = false; std::vector<unsigned char> rgb; int w = 0, h = 0; std::string err; };
    std::future<FrameResult> frameFut;
    std::vector<unsigned char> frameRgb;
    int frameW = 0, frameH = 0;
    std::string frameErr;
    unsigned int frameTex = 0;
    bool frameDirty = false;

    // processing
    float startSec = 0, endSec = 0;
    int targetFrames = 90;
    bool framesAuto = true;      // follow the preset's frames-per-second rule
    int viewsIdx = 0;            // 0 -> 8, 1 -> 14
    float crop[4] = {0, 0, 0, 0};   // top, bottom, left, right
    int matchingIdx = 0;         // sequential, exhaustive, vocab_tree
    // camera solve (optional engine packs "glomap" / "vocab-tree")
    int solverIdx = 0;           // 0 COLMAP incremental, 1 GLOMAP global
    bool loopDetection = false;
    int refineIdx = 1;           // thorough, balanced, fast
    bool gpuBA = false;
    void drawCameraSolve(Recon360PanelHost& host);
    void fetchCatalog(Recon360PanelHost& host);
    void pollCatalog();
    void pollEngineJob(Recon360PanelHost& host);   // re-checks packs + doctor when an install finished
    bool catalogAutoTried = false;
    // Optional packs the chosen settings need that are not installed yet (by catalog id).
    std::vector<std::string> neededPacks() const;
    // Queues the download of neededPacks() ahead of the reconstruction (same FIFO job queue).
    void queueNeededPacks(Recon360PanelHost& host);
    bool packOptionRow(Recon360PanelHost& host, const char* packId, bool installed, bool wanted);

    // masking (optional add-on)
    bool maskEnabled = false;
    bool maskPeople = true, maskVehicles = false, maskAnimals = false, maskSky = false;
    int maskDilate = 8;
    struct MaskResult { bool ok = false; std::vector<unsigned char> rgb; float masked = 0; std::string err, key; };
    std::future<MaskResult> maskFut;
    std::vector<unsigned char> maskRgb;   // overlay for the current preview frame
    std::string maskKey;                  // groups/dilate the overlay was computed with
    std::string maskErr;
    float maskFraction = 0;
    bool showMaskOverlay = false;
    std::string maskGroups() const;
    void drawMasking(Recon360PanelHost& host);

    // quality
    std::vector<Recon360Preset> presets;
    std::string presetErr;
    int presetIdx = 1;           // preview, balanced, high, custom(=3)
    int methodIdx = 0;           // splatfacto, splatfacto-big
    int iterations = 30000, downscaleIdx = 1, shDegree = 3, seed = 42;
    bool scaleReg = false;
    bool useCullAlpha = false, useDensify = false, useCullScale = false;
    float cullAlpha = 0.1f, densifyGrad = 0.0008f, cullScale = 0.5f;
    bool preprocessOnly = false;

    // engine packs (downloadable engines, EnginePacks.h)
    void drawEngines(Recon360PanelHost& host);
    void refreshPackStatus();
    struct CatalogResult { bool ok = false; EngineCatalog cat; std::string err; };
    std::future<CatalogResult> catalogFut;
    EngineCatalog catalog;
    bool catalogOk = false;
    std::string catalogErr;
    std::vector<EnginePackStatus> packStatus;   // parallel to catalog.packs
    std::vector<char> packSel;                  // parallel to catalog.packs
    std::shared_ptr<ConvertJob> engineJob;
    bool engineConfirmOpen = false;
    char catalogUrlBuf[512] = {};
    bool catalogUrlLoaded = false;

    // environment
    Recon360Doctor doctor;
    std::future<Recon360Doctor> doctorFut;
    bool doctorDeep = false;
    char pathBufs[6][1024] = {};
    bool pathBufsLoaded = false;

    // output
    char projectBuf[256] = {};
    char parentBuf[1024] = {};
    char plyBuf[256] = {};
    bool autoLoad = true;

    // running job
    std::shared_ptr<ConvertJob> job;
    double jobStart = 0, jobEnd = 0;
};

} // namespace pf
