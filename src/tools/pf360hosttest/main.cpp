// pf360hosttest — host-side tests for the 360 Reconstruction Studio bridge
// (src/viewer/Recon360.cpp): view-count port, manifest writing, worker process
// launch + JSONL event parsing, video probe, cancellation (process-tree
// teardown) and Z-up output detection.
//
// Needs: the pf360 worker beside the exe (copied by the pfview post-build step)
// or in the source tree, a Python 3.8+ interpreter, and ffmpeg/ffprobe on PATH
// (used to synthesize test videos). Tests that need ffmpeg are skipped with a
// message when it is missing. No GPU / Nerfstudio is needed: jobs stop after
// frame extraction.
#include "viewer/Recon360.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace fs = std::filesystem;
using namespace pf;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                                              \
    do {                                                                                         \
        if (cond) ++g_pass;                                                                      \
        else { ++g_fail; std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

static bool haveFfmpeg() { return std::system("ffmpeg -hide_banner -version >NUL 2>&1") == 0; }

static bool makeVideo(const fs::path& out, int seconds, int w, int h) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -v error -y -f lavfi -i testsrc2=size=%dx%d:rate=30:duration=%d -pix_fmt yuv420p \"%s\"",
                  w, h, seconds, out.string().c_str());
    return std::system(cmd) == 0 && fs::exists(out);
}

#ifdef _WIN32
static int countProcesses(const wchar_t* exeName) {
    int n = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    if (Process32FirstW(snap, &pe)) {
        do { if (_wcsicmp(pe.szExeFile, exeName) == 0) ++n; } while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
    return n;
}
#endif

static void testViews() {
    CHECK(recon360ViewsPerFrame(8, 0, 0, 0, 0) == 8);
    CHECK(recon360ViewsPerFrame(14, 0, 0, 0, 0) == 14);
    CHECK(recon360ViewsPerFrame(8, 0, 0, 0.25f, 0.25f) == 4);
    CHECK(recon360ViewsPerFrame(8, 0, 0.45f, 0, 0) == 6);
    CHECK(recon360ViewsPerFrame(8, 0.45f, 0.45f, 0, 0) == -1);   // Nerfstudio would crash
    CHECK(recon360ViewsPerFrame(14, 0.1f, 0.3f, 0, 0) == -1);
    CHECK(recon360PerspectiveSize(5760, 2880, 8) == 1440);
    CHECK(recon360PerspectiveSize(5760, 2880, 14) == 1088);
    bool unsup = false;
    auto d = recon360ViewDirections(8, 0, 0, 0, 0, unsup);
    CHECK(!unsup && d.size() == 8 && d[0].second == 0.0f);         // middle ring first (Nerfstudio order)
}

static void testPresets() {
    std::string err;
    auto p = recon360LoadPresets(err);
    CHECK(err.empty());
    CHECK(p.size() == 3);
    if (p.size() == 3) {
        CHECK(p[0].key == "preview" && p[2].method == "splatfacto-big");
        CHECK(recon360FramesForWindow(p[1], 30.0) == 90);
        CHECK(recon360FramesForWindow(p[1], 1.0) == p[1].minFrames);
    }
}

static Recon360JobSpec baseSpec(const fs::path& video, const fs::path& out, const char* id) {
    Recon360JobSpec s;
    s.jobId = id;
    s.videoPath = video.u8string();
    s.outputDir = out.u8string();
    s.projectionConfirmed = true;
    s.targetFrames = 8;
    s.qualityPreset = "preview";
    s.method = "splatfacto";
    s.stopAfter = "frame_extraction";
    return s;
}

static void testManifest(const fs::path& tmp) {
    Recon360Settings st;
    Recon360JobSpec sp = baseSpec(tmp / u8"vidéo test.mp4", tmp / "out", "hosttest_manifest");
    sp.cropBottom = 0.2f;
    sp.endSeconds = -1;
    std::string path, err;
    CHECK(recon360WriteManifest(sp, st, path, err));
    std::ifstream f(fs::u8path(path));
    auto j = nlohmann::json::parse(f);
    CHECK(j["schemaVersion"] == 1);
    CHECK(j["jobId"] == "hosttest_manifest");
    CHECK(j["input"]["endTimeSeconds"].is_null());
    CHECK(j["input"]["videoPath"].get<std::string>() == sp.videoPath);   // UTF-8 round trip
    CHECK(std::abs(j["processing"]["cropBottom"].get<double>() - 0.2) < 1e-6);
    CHECK(j["options"]["stopAfter"] == "frame_extraction");
    CHECK(!j["training"].contains("cullAlphaThresh"));                     // unset custom fields omitted
    CHECK(j["processing"]["masking"]["enabled"] == false);                 // masking off by default

    // Masking fields: groups in a fixed order, dilation clamped, enabled only with >= 1 group.
    sp.jobId = "hosttest_manifest_mask";
    sp.maskEnabled = true;
    sp.maskPeople = true; sp.maskVehicles = false; sp.maskAnimals = false; sp.maskSky = true;
    sp.maskDilatePx = 500;
    CHECK(recon360WriteManifest(sp, st, path, err));
    std::ifstream f2(fs::u8path(path));
    auto j2 = nlohmann::json::parse(f2);
    CHECK(j2["processing"]["masking"]["enabled"] == true);
    CHECK(j2["processing"]["masking"]["groups"] == nlohmann::json::array({"people", "sky"}));
    CHECK(j2["processing"]["masking"]["dilatePx"] == 64);
    sp.maskPeople = sp.maskSky = false;
    CHECK(recon360WriteManifest(sp, st, path, err));
    std::ifstream f3(fs::u8path(path));
    CHECK(nlohmann::json::parse(f3)["processing"]["masking"]["enabled"] == false);
}

// Mask preview through the real worker + segmentation model; skipped (not failed)
// when the optional masking add-on is not installed on this machine.
static void testMaskPreview() {
    Recon360Settings st;
    const int w = 256, h = 128;
    std::vector<unsigned char> rgb((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            unsigned char* p = &rgb[((size_t)y * w + x) * 3];
            const bool sky = y < h / 2;          // blue top half, grey bottom half
            p[0] = sky ? 110 : 120; p[1] = sky ? 160 : 120; p[2] = sky ? 230 : 120;
        }
    std::vector<unsigned char> ov;
    float frac = -1;
    std::string err;
    const bool ok = recon360MaskPreview(st, rgb, w, h, "sky", 0, ov, frac, err);
    if (!ok && err.find("not installed") != std::string::npos) {
        std::printf("pf360hosttest: SKIPPED mask preview (%s)\n", err.c_str());
        return;
    }
    if (!ok) std::fprintf(stderr, "mask preview error: %s\n", err.c_str());
    CHECK(ok);
    CHECK(ov.size() == rgb.size());
    CHECK(frac >= 0.0f && frac <= 1.0f);
    std::vector<unsigned char> dummy;
    CHECK(!recon360MaskPreview(st, rgb, w, h, "people;rm -rf", 0, dummy, frac, err));   // groups validated
    std::fprintf(stderr, "mask preview: %.1f%% masked as sky\n", frac * 100.0f);
}

static void testProbe(const fs::path& video) {
    Recon360Settings st;
    Recon360Probe p = recon360ProbeVideo(st, video.u8string(), 0, -1);
    if (!p.ok) std::fprintf(stderr, "probe error: %s\n", p.error.c_str());
    CHECK(p.ok);
    CHECK(p.width == 2048 && p.height == 1024);
    CHECK(p.projection == "likely_equirectangular" && p.requiresConfirmation && p.supported);
    Recon360Probe bad = recon360ProbeVideo(st, (video.parent_path() / "missing.mp4").u8string(), 0, -1);
    CHECK(!bad.ok);
}

static void testRunFramesOnly(const fs::path& tmp, const fs::path& video) {
    Recon360Settings st;
    Recon360JobSpec sp = baseSpec(video, tmp / u8"out ü 1", "hosttest_run");
    Recon360Live live;
    std::atomic<bool> cancel{false};
    float lastPct = -5;
    int progressCalls = 0;
    std::string err;
    const bool ok = recon360Run(sp, st, cancel, live, [&](float p, const std::string&) { lastPct = p; ++progressCalls; }, err);
    if (!ok) std::fprintf(stderr, "run error: %s\n", err.c_str());
    CHECK(ok);
    CHECK(progressCalls > 0);
    CHECK(live.stage[0] == Recon360Live::StageState::Done);
    CHECK(live.stage[1] == Recon360Live::StageState::Done);
    CHECK(live.stage[2] == Recon360Live::StageState::Waiting);
    CHECK(fs::exists(tmp / u8"out ü 1" / "frames" / "frame_00008.jpg"));
    CHECK(!live.reportJson.empty());
    // Second run reuses the cached frame stage.
    Recon360Live live2;
    CHECK(recon360Run(sp, st, cancel, live2, nullptr, err));
    CHECK(live2.stage[1] == Recon360Live::StageState::Skipped);
}

static void testFailurePropagation(const fs::path& tmp, const fs::path& video) {
    Recon360Settings st;
    Recon360JobSpec sp = baseSpec(video, tmp / "out_fail", "hosttest_fail");
    sp.projectionConfirmed = false;          // 2:1 without metadata must be confirmed
    Recon360Live live;
    std::atomic<bool> cancel{false};
    std::string err;
    CHECK(!recon360Run(sp, st, cancel, live, nullptr, err));
    CHECK(err.find("onfirm") != std::string::npos);
    CHECK(live.stage[0] == Recon360Live::StageState::Failed);
}

static void testCancel(const fs::path& tmp, const fs::path& longVideo) {
    Recon360Settings st;
    Recon360JobSpec sp = baseSpec(longVideo, tmp / "out_cancel", "hosttest_cancel");
    sp.targetFrames = 1800;                  // long enough to still be extracting when we cancel
    Recon360Live live;
    std::atomic<bool> cancel{false};
    std::string err;
    std::thread canceller([&] {
        for (int i = 0; i < 600; ++i) {      // wait until frame extraction runs, then cancel
            {
                std::lock_guard<std::mutex> lk(live.mu);
                if (live.currentStage == 1) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        cancel = true;
    });
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = recon360Run(sp, st, cancel, live, nullptr, err);
    canceller.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(!ok);
    CHECK(err == "Canceled");
    CHECK(secs < 60);
#ifdef _WIN32
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CHECK(countProcesses(L"ffmpeg.exe") == 0);   // the whole tree is gone
#endif
    std::fprintf(stderr, "cancel took %.1fs; last message: %s\n", secs, live.finalMessage.c_str());
}

static void testZUpDetection(const fs::path& tmp) {
    const fs::path ws = tmp / "ws_zup";
    fs::create_directories(ws / "export");
    const fs::path ply = ws / "export" / "scene.ply";
    std::ofstream(ply) << "ply\n";
    {
        nlohmann::json r;
        r["output"] = {{"path", ply.u8string()}, {"upAxis", "z_up"}};
        std::ofstream(ws / "reconstruction_report.json") << r.dump();
    }
    CHECK(recon360IsZUpOutput(ply.u8string()));
    CHECK(!recon360IsZUpOutput((tmp / "elsewhere.ply").u8string()));
}

int main() {
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path() / "pf360hosttest";
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);
    std::printf("pf360hosttest: worker root = %s\n", recon360WorkerRoot().c_str());
    std::printf("pf360hosttest: python      = %s\n", recon360FindPython({}).c_str());

    testViews();
    testPresets();
    testManifest(tmp);
    testZUpDetection(tmp);
    if (haveFfmpeg() && !recon360FindPython({}).empty()) {
        const fs::path video = tmp / u8"clip ü.mp4";
        const fs::path longVideo = tmp / "long.mp4";
        CHECK(makeVideo(video, 6, 2048, 1024));
        CHECK(makeVideo(longVideo, 60, 2048, 1024));
        testProbe(video);
        testRunFramesOnly(tmp, video);
        testFailurePropagation(tmp, video);
        testCancel(tmp, longVideo);
        testMaskPreview();
    } else {
        std::printf("pf360hosttest: SKIPPED process tests (need ffmpeg on PATH and Python 3)\n");
    }
    std::printf("pf360hosttest: %d passed, %d failed -> %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
