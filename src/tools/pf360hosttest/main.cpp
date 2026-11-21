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
#include "viewer/EnginePacks.h"
#include "viewer/Recon360.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
#include <tlhelp32.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")
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

// Batch mode: folder scan rules, workspace planning, and several videos run back to back
// with ids that differ only by their batch index (all queued within the same second).
static void testBatch(const fs::path& tmp, const fs::path& video) {
    std::error_code ec;
    const fs::path root = tmp / u8"batch root ü";
    fs::remove_all(root, ec);
    auto put = [&](const fs::path& rel) {
        fs::create_directories((root / rel).parent_path());
        fs::copy_file(video, root / rel, fs::copy_options::overwrite_existing);
    };
    put("a.mp4");
    put(fs::u8path("sub/B.MP4"));                       // upper-case extension
    put(fs::u8path("sub/deeper/c.mkv"));
    put(fs::u8path("PointForge360/old/old.mp4"));       // our own output tree: never scanned
    put(fs::u8path("ws/clip.mp4"));                     // inside a 360 workspace: never scanned
    std::ofstream(root / "ws" / "state.json") << "{}";
    std::ofstream(root / "notes.txt") << "not a video";

    bool truncated = false;
    auto rel = [&](const std::vector<std::string>& v) {
        std::vector<std::string> r;
        for (const auto& p : v) r.push_back(fs::u8path(p).lexically_relative(root).generic_u8string());
        return r;
    };
    const auto all = rel(recon360FindVideos(root.u8string(), true, 100, truncated));
    CHECK((all == std::vector<std::string>{"a.mp4", "sub/B.MP4", "sub/deeper/c.mkv"}));
    CHECK(!truncated);
    CHECK((rel(recon360FindVideos(root.u8string(), false, 100, truncated)) == std::vector<std::string>{"a.mp4"}));
    CHECK(recon360FindVideos(root.u8string(), true, 2, truncated).size() == 2 && truncated);
    CHECK(recon360FindVideos((root / "missing").u8string(), true, 100, truncated).empty());

    const fs::path out = tmp / "batch out";
    fs::remove_all(out, ec);
    CHECK(recon360WorkspaceState((out / "a").u8string(), "a_splat.ply") == Recon360WorkspaceState::New);
    fs::create_directories(out / "foreign");
    std::ofstream(out / "foreign" / "x.txt") << "x";
    CHECK(recon360WorkspaceState((out / "foreign").u8string(), "foreign_splat.ply") == Recon360WorkspaceState::Foreign);

    // Two videos back to back, like the batch queue (one job at a time).
    Recon360Settings st;
    std::atomic<bool> cancel{false};
    std::string err;
    const char* ids[2] = {"pf360_20261121_120000_001", "pf360_20261121_120000_002"};
    const fs::path vids[2] = {root / "a.mp4", root / "sub" / "B.MP4"};
    const char* names[2] = {"a", "B"};
    for (int i = 0; i < 2; ++i) {
        Recon360JobSpec sp = baseSpec(vids[i], out / names[i], ids[i]);
        Recon360Live live;
        const bool ok = recon360Run(sp, st, cancel, live, nullptr, err);
        if (!ok) std::fprintf(stderr, "batch run %d: %s\n", i, err.c_str());
        CHECK(ok);
        CHECK(fs::exists(out / names[i] / "frames" / "frame_00008.jpg"));
        // An interrupted / partial batch item resumes on the next batch.
        CHECK(recon360WorkspaceState((out / names[i]).u8string(), std::string(names[i]) + "_splat.ply") ==
              Recon360WorkspaceState::Resume);
    }
    fs::create_directories(out / "a" / "export");
    std::ofstream(out / "a" / "export" / "a_splat.ply") << "ply";
    CHECK(recon360WorkspaceState((out / "a").u8string(), "a_splat.ply") == Recon360WorkspaceState::Done);
    // The worker wrote one manifest per job id (no overwrite between batch items).
    const fs::path jobsDir = fs::path(_wgetenv(L"LOCALAPPDATA")) / "ViitorX" / "PointForge" / "pf360-jobs";
    CHECK(fs::exists(jobsDir / (std::string(ids[0]) + ".json")) && fs::exists(jobsDir / (std::string(ids[1]) + ".json")));
}

// Restore points across app restarts: a job is rebuilt from its saved manifest and
// unfinished workspaces are listed (finished / deleted ones are not).
static void testUnfinished(const fs::path& tmp, const fs::path& video) {
    std::error_code ec;
    Recon360JobSpec sp = baseSpec(video, tmp / u8"resume ws ü", "hosttest_resume_001");
    sp.stopAfter.clear();
    sp.targetFrames = 77; sp.imagesPerEquirect = 14; sp.cropBottom = 0.2f; sp.matchingMethod = "exhaustive";
    sp.maskEnabled = true; sp.maskPeople = true; sp.maskSky = true; sp.maskDilatePx = 12;
    sp.iterations = 12345; sp.downscaleFactor = 4; sp.method = "splatfacto-big"; sp.plyFileName = "x_splat.ply";
    std::string path, err;
    CHECK(recon360WriteManifest(sp, {}, path, err));
    std::ifstream f(fs::u8path(path), std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Recon360JobSpec back;
    CHECK(recon360SpecFromManifest(text, back, err));
    CHECK(back.jobId == sp.jobId && back.videoPath == sp.videoPath && back.outputDir == sp.outputDir);
    CHECK(back.targetFrames == 77 && back.imagesPerEquirect == 14 && std::fabs(back.cropBottom - 0.2f) < 1e-6f);
    CHECK(back.matchingMethod == "exhaustive" && back.maskEnabled && back.maskPeople && back.maskSky && !back.maskVehicles);
    CHECK(back.maskDilatePx == 12 && back.iterations == 12345 && back.downscaleFactor == 4);
    CHECK(back.method == "splatfacto-big" && back.plyFileName == "x_splat.ply" && back.projectionConfirmed);

    auto listed = [&]() {
        for (const auto& u : recon360ListUnfinished()) if (u.spec.jobId == sp.jobId) return true;
        return false;
    };
    fs::remove_all(fs::u8path(sp.outputDir), ec);
    CHECK(!listed());                                         // never started: no workspace
    fs::create_directories(fs::u8path(sp.outputDir));
    std::ofstream(fs::u8path(sp.outputDir) / "state.json")
        << R"({"marker":"pf360-workspace-v1","stages":{"frame_extraction":{"status":"completed","finishedAt":1},)"
           R"("training":{"status":"cancelled","finishedAt":2}}})";
    bool found = false;
    for (const auto& u : recon360ListUnfinished())
        if (u.spec.jobId == sp.jobId) { found = true; CHECK(u.lastStage == "training" && u.lastStatus == "cancelled"); }
    CHECK(found);
    std::ofstream(fs::u8path(sp.outputDir) / "state.json")
        << R"({"marker":"pf360-workspace-v1","stages":{"validation":{"status":"completed","finishedAt":3}}})";
    CHECK(!listed());                                         // finished
    fs::remove_all(fs::u8path(sp.outputDir), ec);
    fs::remove(fs::u8path(path), ec);
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

#ifdef _WIN32
// ---------------------------------------------------------------------------
// Engine packs: signed catalog, resumable download, SHA-256, staging/swap.
// A tiny loopback HTTP server (byte ranges + a deliberately dropped first
// connection) and a throw-away ECDSA key stand in for Hugging Face + the
// release key.
// ---------------------------------------------------------------------------
namespace {

struct TestKey {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_KEY_HANDLE key = nullptr;
    std::string pubHex;
    TestKey() {
        BCryptOpenAlgorithmProvider(&alg, BCRYPT_ECDSA_P256_ALGORITHM, nullptr, 0);
        BCryptGenerateKeyPair(alg, &key, 256, 0);
        BCryptFinalizeKeyPair(key, 0);
        unsigned char blob[sizeof(BCRYPT_ECCKEY_BLOB) + 64];
        ULONG cb = 0;
        BCryptExportKey(key, nullptr, BCRYPT_ECCPUBLIC_BLOB, blob, sizeof(blob), &cb, 0);
        char h[3];
        for (int i = 0; i < 64; ++i) { std::snprintf(h, 3, "%02x", blob[sizeof(BCRYPT_ECCKEY_BLOB) + i]); pubHex += h; }
    }
    ~TestKey() { if (key) BCryptDestroyKey(key); if (alg) BCryptCloseAlgorithmProvider(alg, 0); }
    std::string sign(const std::string& data) const {
        unsigned char digest[32], sig[64];
        BCRYPT_ALG_HANDLE sha = nullptr;
        BCryptOpenAlgorithmProvider(&sha, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        BCryptHash(sha, nullptr, 0, (PUCHAR)data.data(), (ULONG)data.size(), digest, 32);
        BCryptCloseAlgorithmProvider(sha, 0);
        ULONG cb = 0;
        BCryptSignHash(key, nullptr, digest, 32, sig, 64, &cb, 0);
        static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        for (int i = 0; i < 64; i += 3) {
            const unsigned v = (sig[i] << 16) | ((i + 1 < 64 ? sig[i + 1] : 0) << 8) | (i + 2 < 64 ? sig[i + 2] : 0);
            out += t[(v >> 18) & 63];
            out += t[(v >> 12) & 63];
            out += i + 1 < 64 ? t[(v >> 6) & 63] : '=';
            out += i + 2 < 64 ? t[v & 63] : '=';
        }
        return out;
    }
};

struct TestServer {
    SOCKET ls = INVALID_SOCKET;
    int port = 0;
    fs::path dir;
    std::thread th;
    std::mutex mu;
    std::set<std::string> dropOnce;     // paths whose first GET is cut after half the body
    int requests = 0, rangeRequests = 0;

    explicit TestServer(const fs::path& d) : dir(d) {
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
        ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind(ls, (sockaddr*)&a, sizeof(a));
        int len = sizeof(a);
        getsockname(ls, (sockaddr*)&a, &len);
        port = ntohs(a.sin_port);
        listen(ls, 8);
        th = std::thread([this] { loop(); });
    }
    ~TestServer() {
        closesocket(ls);
        if (th.joinable()) th.join();
        WSACleanup();
    }
    std::string url(const std::string& name) const { return "http://127.0.0.1:" + std::to_string(port) + "/" + name; }

    void loop() {
        for (;;) {
            SOCKET c = accept(ls, nullptr, nullptr);
            if (c == INVALID_SOCKET) return;
            serve(c);
            closesocket(c);
        }
    }
    void serve(SOCKET c) {
        std::string req;
        char b[2048];
        while (req.find("\r\n\r\n") == std::string::npos) {
            const int n = recv(c, b, sizeof(b), 0);
            if (n <= 0) return;
            req.append(b, (size_t)n);
        }
        const size_t sp1 = req.find(' '), sp2 = req.find(' ', sp1 + 1);
        std::string path = req.substr(sp1 + 2, sp2 - sp1 - 2);
        uint64_t from = 0;
        bool ranged = false;
        const size_t r = req.find("Range: bytes=");
        if (r != std::string::npos) { from = std::strtoull(req.c_str() + r + 13, nullptr, 10); ranged = true; }
        bool drop = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            ++requests;
            if (ranged) ++rangeRequests;
            if (dropOnce.erase(path)) drop = true;
        }
        const fs::path f = dir / path;
        std::error_code ec;
        if (path.find("..") != std::string::npos || !fs::is_regular_file(f, ec)) {
            const char* nf = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(c, nf, (int)std::strlen(nf), 0);
            return;
        }
        std::ifstream in(f, std::ios::binary);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const uint64_t size = body.size();
        std::string hdr;
        if (ranged && from < size) {
            hdr = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes " + std::to_string(from) + "-" +
                  std::to_string(size - 1) + "/" + std::to_string(size) + "\r\n";
            body = body.substr((size_t)from);
        } else {
            hdr = "HTTP/1.1 200 OK\r\n";
        }
        hdr += "Content-Length: " + std::to_string(body.size()) + "\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n";
        send(c, hdr.data(), (int)hdr.size(), 0);
        const size_t n = drop ? body.size() / 2 : body.size();
        for (size_t off = 0; off < n;) {
            const int s = send(c, body.data() + off, (int)std::min<size_t>(65536, n - off), 0);
            if (s <= 0) break;
            off += (size_t)s;
        }
    }
};

std::string sha256Of(const fs::path& p) {
    std::string h;
    engineSha256File(p.u8string(), h);
    return h;
}

void writeText(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << s;
}

std::string readText(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

} // namespace

static void testEnginePacks(const fs::path& tmp) {
    std::error_code ec;
    const fs::path srv = tmp / "pack-server", root = tmp / u8"engines ü", src = tmp / "pack-src";
    fs::remove_all(srv, ec);
    fs::remove_all(root, ec);
    fs::remove_all(src, ec);
    fs::create_directories(srv);

    // A zip pack (unpacked into engines/testpack) with ~3 MB of incompressible data so the
    // dropped-connection resume path really runs, and a "file" pack into a nested dir.
    std::string big(3u << 20, '\0');
    uint32_t x = 12345;
    for (char& ch : big) { x = x * 1664525u + 1013904223u; ch = (char)(x >> 24); }
    writeText(src / "testpack" / "hello.txt", "hello pack\n");
    writeText(src / "testpack" / "data.bin", big);
    const std::wstring tarCmd = L"tar.exe -a -c -f \"" + (srv / "testpack.zip").wstring() + L"\" -C \"" +
                                src.wstring() + L"\" testpack";
    CHECK(_wsystem(tarCmd.c_str()) == 0);
    writeText(srv / "model.json", "{\"m\":1}\n");

    TestKey key;
    CHECK(key.pubHex.size() == 128);
    auto catalogJson = [&](int64_t seq, const std::string& zipSha) {
        nlohmann::json j = {
            {"schema", 1}, {"sequence", seq}, {"created", "test"},
            {"packs", {
                {{"id", "testpack"}, {"title", "Test pack"}, {"version", "1.0"}, {"installDir", "testpack"},
                 {"postInstall", ""}, {"required", true}, {"installedBytes", big.size() + 11},
                 {"files", {{{"name", "testpack.zip"}, {"kind", "zip"}, {"url", "PLACEHOLDER_ZIP"},
                             {"size", fs::file_size(srv / "testpack.zip")}, {"sha256", zipSha}, {"fileCount", 3}}}}},
                {{"id", "model"}, {"title", "Model"}, {"version", "7"}, {"installDir", "models/m1"},
                 {"postInstall", ""}, {"required", false}, {"installedBytes", 8},
                 {"files", {{{"name", "model.json"}, {"kind", "file"}, {"url", "PLACEHOLDER_MODEL"},
                             {"size", fs::file_size(srv / "model.json")}, {"sha256", sha256Of(srv / "model.json")}}}}}}}};
        return j;
    };
    TestServer server(srv);
    auto publish = [&](nlohmann::json j) {
        j["packs"][0]["files"][0]["url"] = server.url("testpack.zip");
        j["packs"][1]["files"][0]["url"] = server.url("model.json");
        const std::string body = j.dump(1);
        writeText(srv / "catalog.json", body);
        writeText(srv / "catalog.json.sig", key.sign(body) + "\n");
        return body;
    };
    const std::string zipSha = sha256Of(srv / "testpack.zip");

    // --- signature + parser hardening
    {
        const std::string body = publish(catalogJson(100, zipSha));
        std::string err;
        CHECK(engineVerifySignature(body, key.sign(body), key.pubHex, err));
        CHECK(!engineVerifySignature(body + " ", key.sign(body), key.pubHex, err));          // tampered
        TestKey other;
        CHECK(!engineVerifySignature(body, key.sign(body), other.pubHex, err));              // wrong key
        CHECK(!engineVerifySignature(body, "AAAA", key.pubHex, err));                       // malformed
        CHECK(!engineVerifySignature(body, key.sign(body), "", err));                       // no key
        EngineCatalog c;
        auto bad = [&](const char* field, nlohmann::json v) {
            nlohmann::json j = catalogJson(1, zipSha);
            j["packs"][0]["files"][0]["url"] = "https://example.invalid/x.zip";
            j["packs"][1]["files"][0]["url"] = "https://example.invalid/m.json";
            if (std::string(field) == "url") j["packs"][0]["files"][0]["url"] = v;
            else if (std::string(field) == "name") j["packs"][0]["files"][0]["name"] = v;
            else j["packs"][0][field] = v;
            std::string e;
            return !engineParseCatalog(j.dump(), c, e);
        };
        CHECK(bad("installDir", "../escape"));
        CHECK(bad("installDir", "C:/abs"));
        CHECK(bad("postInstall", "cmd /c del"));
        CHECK(bad("url", "ftp://host/x.zip"));
        CHECK(bad("url", "http://example.com/x.zip"));      // plain http only on loopback
        CHECK(bad("name", "..\\x.zip"));
        CHECK(!bad("installDir", "models/ok-dir"));         // control: a valid catalog parses
    }

    // --- fetch + install (first connection for the zip is dropped half way -> resume)
    EngineCatalog cat;
    std::string err;
    CHECK(!engineFetchCatalog(server.url("catalog.json"), root.u8string(), cat, err, "") || !engineHasSigningKey());
    err.clear();
    CHECK(engineFetchCatalog(server.url("catalog.json"), root.u8string(), cat, err, key.pubHex));
    if (!err.empty()) std::fprintf(stderr, "  fetch: %s\n", err.c_str());
    CHECK(cat.packs.size() == 2);
    // A manual setup already sits where the model goes: it must be kept, not deleted.
    writeText(root / "models" / "m1" / "mine.txt", "user data");
    {
        std::lock_guard<std::mutex> lk(server.mu);
        server.dropOnce.insert("testpack.zip");
    }
    std::atomic<bool> cancel{false};
    float lastPct = -1;
    err.clear();
    const bool ok = engineInstallPacks(cat, {"testpack", "model"}, root.u8string(), cancel,
                                       [&](float p, const std::string&) { lastPct = p; }, err);
    if (!ok) std::fprintf(stderr, "  install: %s\n", err.c_str());
    CHECK(ok);
    CHECK(lastPct >= 0.99f);
    CHECK(server.rangeRequests >= 1);                                   // resumed with a byte range
    CHECK(readText(root / "testpack" / "hello.txt") == "hello pack\n");
    CHECK(fs::file_size(root / "testpack" / "data.bin", ec) == big.size());
    CHECK(readText(root / "models" / "m1" / "model.json") == "{\"m\":1}\n");
    CHECK(engineStatus(root.u8string(), cat.packs[0]).state == EnginePackStatus::State::Installed);
    CHECK(engineStatus(root.u8string(), cat.packs[1]).state == EnginePackStatus::State::Installed);
    bool backupKept = false;
    for (const auto& e : fs::directory_iterator(root / "models"))
        if (e.path().filename().u8string().rfind("m1.backup-", 0) == 0 && fs::exists(e.path() / "mine.txt"))
            backupKept = true;
    CHECK(backupKept);
    CHECK(!fs::exists(root / ".downloads" / "testpack"));               // downloads cleaned up
    CHECK(!fs::exists(root / ".staging" / "testpack"));

    // --- update: a newer catalog with a new version replaces the managed install (no .old left)
    {
        nlohmann::json j = catalogJson(200, zipSha);
        j["packs"][0]["version"] = "1.1";
        publish(j);
        EngineCatalog c2;
        CHECK(engineFetchCatalog(server.url("catalog.json"), root.u8string(), c2, err, key.pubHex));
        CHECK(engineStatus(root.u8string(), c2.packs[0]).state == EnginePackStatus::State::UpdateAvailable);
        CHECK(engineInstallPacks(c2, {"testpack"}, root.u8string(), cancel, nullptr, err));
        CHECK(engineStatus(root.u8string(), c2.packs[0]).state == EnginePackStatus::State::Installed);
        CHECK(!fs::exists(root / "testpack.old"));
    }

    // --- rollback protection: an older (validly signed) catalog is refused
    {
        publish(catalogJson(150, zipSha));
        EngineCatalog c3;
        err.clear();
        CHECK(!engineFetchCatalog(server.url("catalog.json"), root.u8string(), c3, err, key.pubHex));
        CHECK(err.find("OLDER") != std::string::npos);
    }

    // --- hash mismatch: nothing is installed, the bad download is deleted, the old install stays
    {
        nlohmann::json j = catalogJson(300, std::string(64, 'a'));
        j["packs"][0]["version"] = "2.0";
        publish(j);
        EngineCatalog c4;
        CHECK(engineFetchCatalog(server.url("catalog.json"), root.u8string(), c4, err, key.pubHex));
        err.clear();
        CHECK(!engineInstallPacks(c4, {"testpack"}, root.u8string(), cancel, nullptr, err));
        CHECK(err.find("SHA-256") != std::string::npos);
        CHECK(!fs::exists(root / ".downloads" / "testpack" / "testpack.zip"));
        CHECK(readText(root / "testpack" / "hello.txt") == "hello pack\n");
        CHECK(engineStatus(root.u8string(), c4.packs[0]).installedVersion == "1.1");
    }

    // --- tampered catalog on the server (signature no longer matches)
    {
        nlohmann::json j = catalogJson(400, zipSha);
        const std::string body = publish(j);
        writeText(srv / "catalog.json", body + "\n");
        EngineCatalog c5;
        err.clear();
        CHECK(!engineFetchCatalog(server.url("catalog.json"), root.u8string(), c5, err, key.pubHex));
        CHECK(err.find("INVALID") != std::string::npos);
    }
}
#endif

// CLI helpers for manual / real-pack checks:
//   pf360hosttest --verify <catalog.json> <pubhex>        (signature written by engine_packs.py)
//   pf360hosttest --install <catalogUrl> <root> <pubhex> [ids...]
static int cliMode(int argc, char** argv) {
    const std::string mode = argv[1];
    if (mode == "--verify" && argc == 4) {
        std::string body = readText(fs::u8path(argv[2])), sig = readText(fs::u8path(std::string(argv[2]) + ".sig")), err;
        const bool ok = engineVerifySignature(body, sig, argv[3], err);
        EngineCatalog c;
        const bool parsed = ok && engineParseCatalog(body, c, err);
        std::printf("signature %s, parse %s %s\n", ok ? "OK" : "INVALID", parsed ? "OK" : "FAILED", err.c_str());
        return ok && parsed ? 0 : 1;
    }
    if (mode == "--install" && argc >= 5) {
        EngineCatalog cat;
        std::string err;
        if (!engineFetchCatalog(argv[2], argv[3], cat, err, argv[4])) { std::printf("fetch failed: %s\n", err.c_str()); return 1; }
        std::vector<std::string> ids;
        for (int i = 5; i < argc; ++i) ids.push_back(argv[i]);
        if (ids.empty()) for (const auto& p : cat.packs) ids.push_back(p.id);
        std::atomic<bool> cancel{false};
        int lastTenth = -1;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = engineInstallPacks(cat, ids, argv[3], cancel, [&](float p, const std::string& m) {
            const int tenth = (int)(p * 20);
            if (tenth != lastTenth) { lastTenth = tenth; std::printf("  %3.0f%%  %s\n", p * 100, m.c_str()); std::fflush(stdout); }
        }, err);
        std::printf("install %s in %.0f s %s\n", ok ? "OK" : "FAILED",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), err.c_str());
        return ok ? 0 : 1;
    }
    std::printf("usage: pf360hosttest [--verify catalog pubhex | --install url root pubhex [ids]]\n");
    return 2;
}

int main(int argc, char** argv) {
    if (argc > 1) return cliMode(argc, argv);
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
#ifdef _WIN32
    testEnginePacks(tmp);
#endif
    if (haveFfmpeg() && !recon360FindPython({}).empty()) {
        const fs::path video = tmp / u8"clip ü.mp4";
        const fs::path longVideo = tmp / "long.mp4";
        CHECK(makeVideo(video, 6, 2048, 1024));
        CHECK(makeVideo(longVideo, 60, 2048, 1024));
        testProbe(video);
        testRunFramesOnly(tmp, video);
        testBatch(tmp, video);
        testUnfinished(tmp, video);
        testFailurePropagation(tmp, video);
        testCancel(tmp, longVideo);
        testMaskPreview();
    } else {
        std::printf("pf360hosttest: SKIPPED process tests (need ffmpeg on PATH and Python 3)\n");
    }
    std::printf("pf360hosttest: %d passed, %d failed -> %s\n", g_pass, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
