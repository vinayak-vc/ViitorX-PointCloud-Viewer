#include "Recon360.h"
#include "common/Log.h"
#include "io/SplatReader.h"

#include <nlohmann/json.hpp>

#ifdef PF_EMBED_WORKER
#include "EmbeddedWorker.h"   // generated: build/generated (tools/embed_worker.cmake)
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <tlhelp32.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace pf {

static std::atomic<bool> g_recon360ShuttingDown{false};
void recon360SetShuttingDown() { g_recon360ShuttingDown = true; }

// ---------------------------------------------------------------------------
// stage labels / weights
// ---------------------------------------------------------------------------
const char* recon360StageLabel(int i) {
    constexpr int n = Recon360Live::kStageCount;
    static const char* labels[n] = {"Video validation", "Frame extraction", "Panorama projection",
                                    "Masking (people / vehicles / sky)", "Camera reconstruction (COLMAP)",
                                    "Gaussian training", "Gaussian export", "PLY validation"};
    return (i >= 0 && i < n) ? labels[i] : "Preparing";
}

// ---------------------------------------------------------------------------
// batch helpers (portable std::filesystem)
// ---------------------------------------------------------------------------
namespace {
bool isVideoExt(const fs::path& p) {
    std::string e = p.extension().u8string();
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    return e == ".mp4" || e == ".mov" || e == ".mkv";
}
bool isOwnOutputDir(const fs::path& d) {
    std::error_code ec;
    return d.filename() == "PointForge360" || fs::exists(d / "state.json", ec) ||
           fs::exists(d / "reconstruction_report.json", ec);
}
} // namespace

std::vector<std::string> recon360FindVideos(const std::string& rootUtf8, bool recursive, size_t maxCount,
                                            bool& truncated, const std::atomic<bool>* cancel) {
    truncated = false;
    std::vector<std::string> out;
    std::error_code ec;
    const fs::path root = fs::u8path(rootUtf8);
    if (!fs::is_directory(root, ec)) return out;
    auto add = [&](const fs::path& p) {
        if (out.size() >= maxCount) { truncated = true; return; }
        out.push_back(p.u8string());
    };
    if (recursive) {
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; !ec && it != end && !truncated && !(cancel && cancel->load()); it.increment(ec)) {
            std::error_code e2;
            if (it->is_directory(e2)) {
                if (isOwnOutputDir(it->path())) it.disable_recursion_pending();
            } else if (it->is_regular_file(e2) && isVideoExt(it->path())) {
                add(it->path());
            }
        }
    } else {
        for (const auto& e : fs::directory_iterator(root, fs::directory_options::skip_permission_denied, ec)) {
            std::error_code e2;
            if (e.is_regular_file(e2) && isVideoExt(e.path())) add(e.path());
            if (truncated) break;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

Recon360WorkspaceState recon360WorkspaceState(const std::string& dirUtf8, const std::string& plyName) {
    std::error_code ec;
    const fs::path dir = fs::u8path(dirUtf8);
    if (fs::exists(dir / "export" / fs::u8path(plyName), ec)) return Recon360WorkspaceState::Done;
    if (fs::exists(dir / "state.json", ec)) return Recon360WorkspaceState::Resume;
    if (fs::exists(dir, ec) && !fs::is_empty(dir, ec)) return Recon360WorkspaceState::Foreign;
    return Recon360WorkspaceState::New;
}

float recon360Overall(int stage, float pct) {
    // Rough share of wall-clock per stage: only used to place the overall bar;
    // the per-stage percent shown next to it is the backend's real value.
    constexpr int n = Recon360Live::kStageCount;
    static const float w[n] = {0.02f, 0.05f, 0.07f, 0.03f, 0.19f, 0.60f, 0.02f, 0.02f};
    if (stage < 0) return 0.0f;
    float acc = 0;
    for (int i = 0; i < stage && i < n; ++i) acc += w[i];
    if (stage < n && pct >= 0) acc += w[stage] * std::min(100.0f, pct) / 100.0f;
    return std::min(1.0f, acc);
}

// ---------------------------------------------------------------------------
// view geometry (port of pf360/views.py == Nerfstudio 1.1.5 equirect_utils)
// ---------------------------------------------------------------------------
std::vector<std::pair<float, float>> recon360ViewDirections(int ipe, float top, float bottom, float left,
                                                            float right, bool& unsupported) {
    unsupported = false;
    std::vector<std::pair<float, float>> out;
    if (ipe != 8 && ipe != 14) return out;
    const double fov = ipe == 8 ? 120.0 : 110.0;
    const double steps[3] = {ipe == 8 ? 180.0 : 90.0, ipe == 8 ? 90.0 : 60.0, ipe == 8 ? 180.0 : 90.0};
    double lb = -180, rb = 180;
    if (right > 0) lb = -180 + 360.0 * right;
    if (left > 0)  rb = 180 - 360.0 * left;
    double b[3] = {-45, 0, 45};
    bool alive[3] = {true, true, true};
    if (bottom > 0) {
        const double start = 90 - 180.0 * bottom - fov / 2;
        for (int i = 2; i >= 0; --i) {
            if (b[i] > start + fov / 2) { alive[i] = false; continue; }
            if (b[i] > start) {
                const double diff = b[i] - start;
                b[i] = start;
                for (int j = i - 1; j >= 0; --j) b[j] -= diff / std::pow(2.0, i - j);
                break;
            }
        }
    }
    if (top > 0) {
        const double start = -90 + 180.0 * top + fov / 2;
        for (int i = 0; i < 3; ++i) {
            if (!alive[i]) { unsupported = true; return {}; }      // Nerfstudio: None < float
            if (b[i] < start - fov / 2) { alive[i] = false; continue; }
            if (b[i] < start) {
                const double diff = start - b[i];
                b[i] = start;
                for (int j = i + 1; j < 3; ++j) {
                    if (!alive[j]) { unsupported = true; return {}; }  // Nerfstudio: None += float
                    b[j] += diff / std::pow(2.0, j - i);
                }
                break;
            }
        }
    }
    for (int ring : {1, 2, 0}) {
        if (!alive[ring]) continue;
        const int n = std::max(0, (int)std::ceil((rb - lb) / steps[ring]));
        for (int k = 0; k < n; ++k) out.push_back({(float)(lb + k * steps[ring]), (float)b[ring]});
    }
    return out;
}

int recon360ViewsPerFrame(int ipe, float top, float bottom, float left, float right) {
    bool unsupported = false;
    auto v = recon360ViewDirections(ipe, top, bottom, left, right, unsupported);
    return unsupported ? -1 : (int)v.size();
}

int recon360PerspectiveSize(int w, int h, int ipe) {
    if (w <= 0 || h <= 0 || ipe <= 0) return 0;
    return (int)std::sqrt((double)w * (double)h / (double)ipe);
}

#ifdef _WIN32
// ---------------------------------------------------------------------------
// UTF-8 <-> UTF-16 + argument quoting (MSVCRT CommandLineToArgvW rules)
// ---------------------------------------------------------------------------
namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring quoteArg(const std::wstring& a) {
    return recon360QuoteArg(a);
}

fs::path exeDir() {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof(buf) / sizeof(buf[0])));
    return fs::path(std::wstring(buf, n)).parent_path();
}

fs::path localAppData() {
    const wchar_t* v = _wgetenv(L"LOCALAPPDATA");
    if (!v || !*v) v = _wgetenv(L"TEMP");
    return fs::path(v ? v : L".");
}

bool isFile(const std::wstring& p) {
    std::error_code ec;
    return !p.empty() && fs::is_regular_file(fs::path(p), ec);
}

std::wstring searchPath(const wchar_t* name) {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = SearchPathW(nullptr, name, L".exe", (DWORD)(sizeof(buf) / sizeof(buf[0])), buf, nullptr);
    return n ? std::wstring(buf, n) : std::wstring();
}

// Child environment: current env + UTF-8 Python + worker on PYTHONPATH +
// explicit tool overrides (read by pf360.environment.resolve_tools).
std::vector<wchar_t> buildEnv(const Recon360Settings& s, const std::wstring& workerRoot) {
    std::vector<std::pair<std::wstring, std::wstring>> extra = {
        {L"PYTHONUTF8", L"1"}, {L"PYTHONIOENCODING", L"utf-8"}, {L"PYTHONUNBUFFERED", L"1"},
        {L"PYTHONPATH", workerRoot}};
    if (!s.nsPython.empty())   extra.push_back({L"PF360_NS_PYTHON", widen(s.nsPython)});
    if (!s.colmapExe.empty())  extra.push_back({L"PF360_COLMAP", widen(s.colmapExe)});
    if (!s.ffmpegExe.empty())  extra.push_back({L"PF360_FFMPEG", widen(s.ffmpegExe)});
    if (!s.ffprobeExe.empty()) extra.push_back({L"PF360_FFPROBE", widen(s.ffprobeExe)});
    if (!s.maskModelDir.empty()) extra.push_back({L"PF360_MASK_MODEL", widen(s.maskModelDir)});
    if (!s.lichtfeldExe.empty()) extra.push_back({L"PF360_LICHTFELD", widen(s.lichtfeldExe)});
    std::vector<wchar_t> block;
    LPWCH cur = GetEnvironmentStringsW();
    for (LPWCH p = cur; p && *p;) {
        std::wstring e(p);
        p += e.size() + 1;
        const size_t eq = e.find(L'=', 1);
        const std::wstring key = eq == std::wstring::npos ? e : e.substr(0, eq);
        bool overridden = false;
        for (const auto& kv : extra)
            if (_wcsicmp(kv.first.c_str(), key.c_str()) == 0) overridden = true;
        if (overridden) continue;
        block.insert(block.end(), e.begin(), e.end());
        block.push_back(L'\0');
    }
    if (cur) FreeEnvironmentStringsW(cur);
    for (const auto& kv : extra) {
        std::wstring e = kv.first + L"=" + kv.second;
        block.insert(block.end(), e.begin(), e.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

// A child process tree confined to a Job Object (KILL_ON_JOB_CLOSE).
struct Child {
    HANDLE process = nullptr, job = nullptr, outRd = nullptr, errRd = nullptr, inWr = nullptr;
    ~Child() { close(); }
    void close() {
        for (HANDLE* h : {&inWr, &outRd, &errRd, &process}) if (*h) { CloseHandle(*h); *h = nullptr; }
        if (job) { CloseHandle(job); job = nullptr; }   // kills anything still alive in the job
    }
    void terminateTree() { if (job) TerminateJobObject(job, 1); else if (process) TerminateProcess(process, 1); }
};

bool spawn(const std::vector<std::wstring>& argv, const std::wstring& cwd, const std::vector<wchar_t>& env,
           bool wantStdin, Child& c, std::string& err) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE outWr = nullptr, errWr = nullptr, inRd = nullptr;
    if (!CreatePipe(&c.outRd, &outWr, &sa, 0) || !CreatePipe(&c.errRd, &errWr, &sa, 0)) {
        err = "CreatePipe failed";
        return false;
    }
    SetHandleInformation(c.outRd, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(c.errRd, HANDLE_FLAG_INHERIT, 0);
    if (wantStdin) {
        if (!CreatePipe(&inRd, &c.inWr, &sa, 0)) { err = "CreatePipe failed"; return false; }
        SetHandleInformation(c.inWr, HANDLE_FLAG_INHERIT, 0);
    }
    c.job = CreateJobObjectW(nullptr, nullptr);
    if (c.job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(c.job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = outWr;
    si.hStdError = errWr;
    si.hStdInput = wantStdin ? inRd : nullptr;
    std::wstring cmd;
    for (size_t i = 0; i < argv.size(); ++i) { if (i) cmd += L' '; cmd += quoteArg(argv[i]); }
    std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back(L'\0');
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(argv[0].c_str(), cmdBuf.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED,
                                   (LPVOID)env.data(), cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    CloseHandle(outWr);
    CloseHandle(errWr);
    if (inRd) CloseHandle(inRd);
    if (!ok) {
        err = "could not start " + narrow(argv[0]) + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    if (c.job && !AssignProcessToJobObject(c.job, pi.hProcess))
        logWarn("[pf360] AssignProcessToJobObject failed; child tree cleanup falls back to the worker");
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    c.process = pi.hProcess;
    return true;
}

// Reads a pipe to EOF on its own thread, line by line.
std::thread lineReader(HANDLE h, std::function<void(const std::string&)> onLine) {
    return std::thread([h, onLine]() {
        std::string pending;
        char buf[8192];
        DWORD got = 0;
        while (ReadFile(h, buf, sizeof(buf), &got, nullptr) && got > 0) {
            pending.append(buf, got);
            size_t nl;
            while ((nl = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                while (!line.empty() && (line.back() == '\r')) line.pop_back();
                if (!line.empty()) onLine(line);
            }
            if (pending.size() > (1u << 20)) { onLine(pending); pending.clear(); }
        }
        if (!pending.empty()) onLine(pending);
    });
}

std::vector<std::wstring> pythonLauncher(const Recon360Settings& s) {
    if (isFile(widen(s.workerPython))) return {widen(s.workerPython)};
    // The worker is stdlib-only: any CPython >= 3.8 works, including the Nerfstudio env's.
    const fs::path envDir = localAppData() / L"ViitorX" / L"PointForge" / L"engines" / L"pf360-env";
    const std::wstring envPy = (envDir / L"Scripts" / L"python.exe").wstring();   // developer venv
    const std::wstring packPy = (envDir / L"python.exe").wstring();               // downloaded runtime pack
    // Prefer PointForge's own interpreters (known version, no surprises from
    // whatever Python the user installed), then the user's.
    if (isFile(packPy)) return {packPy};
    if (isFile(envPy)) return {envPy};
    if (isFile(widen(s.nsPython))) return {widen(s.nsPython)};
    std::wstring py = searchPath(L"py");
    if (!py.empty()) return {py, L"-3"};
    py = searchPath(L"python");
    // Skip the Microsoft Store alias stub (it opens the Store instead of running).
    if (!py.empty() && py.find(L"WindowsApps") == std::wstring::npos) return {py};
    return {};
}

// Runs `python -m pf360 <args>` and returns stdout (bounded) — for short commands.
bool runWorkerCapture(const Recon360Settings& s, const std::vector<std::wstring>& args, int timeoutMs,
                      std::string& out, std::string& err) {
    const std::string root = recon360WorkerRoot();
    if (root.empty()) { err = "pf360 worker files not found next to PointForge (pf360\\pf360\\__main__.py)"; return false; }
    std::vector<std::wstring> argv = pythonLauncher(s);
    if (argv.empty()) { err = "No Python 3 interpreter found. Install Python 3.8+ or set one in Preferences > 360 Reconstruction."; return false; }
    argv.push_back(L"-m");
    argv.push_back(L"pf360");
    argv.insert(argv.end(), args.begin(), args.end());
    Child c;
    if (!spawn(argv, widen(root), buildEnv(s, widen(root)), false, c, err)) return false;
    std::string errText;
    std::mutex m;
    std::thread te = lineReader(c.errRd, [&](const std::string& l) {
        std::lock_guard<std::mutex> lk(m);
        if (errText.size() < 16384) errText += l + "\n";
    });
    std::thread to = lineReader(c.outRd, [&](const std::string& l) {
        if (out.size() < (8u << 20)) out += l + "\n";
    });
    const DWORD w = WaitForSingleObject(c.process, (DWORD)timeoutMs);
    c.terminateTree();      // timeout, or stray grandchildren still holding the pipes
    WaitForSingleObject(c.process, 5000);
    to.join();
    te.join();
    DWORD code = 1;
    GetExitCodeProcess(c.process, &code);
    if (w != WAIT_OBJECT_0) { err = "worker timed out"; return false; }
    if (out.empty()) {
        err = errText.empty() ? "worker produced no output (exit " + std::to_string(code) + ")" : errText.substr(0, 600);
        return false;
    }
    return true;
}

std::string str(const json& j, const char* k) {
    auto it = j.find(k);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::string();
}
double num(const json& j, const char* k, double def = 0) {
    auto it = j.find(k);
    return (it != j.end() && it->is_number()) ? it->get<double>() : def;
}
bool boolean(const json& j, const char* k) {
    auto it = j.find(k);
    return it != j.end() && it->is_boolean() && it->get<bool>();
}
std::vector<std::string> strList(const json& j, const char* k) {
    std::vector<std::string> v;
    auto it = j.find(k);
    if (it != j.end() && it->is_array())
        for (const auto& e : *it) if (e.is_string()) v.push_back(e.get<std::string>());
    return v;
}

} // namespace

#ifdef PF_EMBED_WORKER
// The worker compiled into the exe (tools/embed_worker.cmake). Extracted once per
// worker version to %LOCALAPPDATA%\ViitorX\PointForge\worker\<hash>\pf360\...; a
// ".complete" marker is written last, and extraction goes through a temp folder +
// rename, so a crash mid-way never leaves a half-written worker in use.
static std::string extractEmbeddedWorker() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    std::error_code ec;
    const fs::path base = localAppData() / L"ViitorX" / L"PointForge" / L"worker";
    const fs::path dir = base / widen(pf_embedded_worker::kHash);
    if (fs::exists(dir / L".complete", ec)) return narrow(dir.wstring());
    const fs::path tmp = base / (widen(pf_embedded_worker::kHash) + L".tmp" + std::to_wstring(GetCurrentProcessId()));
    fs::remove_all(tmp, ec);
    for (size_t i = 0; i < pf_embedded_worker::kFileCount; ++i) {
        const auto& f = pf_embedded_worker::kFiles[i];
        const fs::path out = tmp / L"pf360" / fs::u8path(f.path);
        fs::create_directories(out.parent_path(), ec);
        std::ofstream o(out, std::ios::binary | std::ios::trunc);
        o.write(reinterpret_cast<const char*>(f.data), (std::streamsize)f.size);
        if (!o) { logError("[pf360] cannot extract the built-in worker to " + narrow(tmp.wstring())); fs::remove_all(tmp, ec); return {}; }
    }
    std::ofstream(tmp / L".complete") << "ok\n";
    fs::remove_all(dir, ec);              // an incomplete earlier attempt
    fs::rename(tmp, dir, ec);
    if (ec) {                             // another PointForge instance won the race
        fs::remove_all(tmp, ec);
        return fs::exists(dir / L".complete", ec) ? narrow(dir.wstring()) : std::string();
    }
    logInfo("[pf360] built-in worker extracted to " + narrow(dir.wstring()));
    return narrow(dir.wstring());
}
#endif

std::string recon360WorkerRoot() {
    std::error_code ec;
    const fs::path exe = exeDir();
    if (fs::exists(exe / L"pf360" / L"pf360" / L"__main__.py", ec)) return narrow((exe / L"pf360").wstring());
    // Dev builds (build\Release\...): walk up to the repo's tools/pf360.
    fs::path p = exe;
    for (int i = 0; i < 5 && !p.empty(); ++i, p = p.parent_path())
        if (fs::exists(p / L"tools" / L"pf360" / L"pf360" / L"__main__.py", ec))
            return narrow((p / L"tools" / L"pf360").wstring());
#ifdef PF_EMBED_WORKER
    // A lone exe (copied to another PC / folder): use the copy built into it.
    static std::string embedded;          // extracted once per process
    if (embedded.empty() || !fs::exists(fs::u8path(embedded) / "pf360" / "__main__.py", ec))
        embedded = extractEmbeddedWorker();
    return embedded;
#else
    return {};
#endif
}

std::string recon360FindPython(const Recon360Settings& s) {
    auto v = pythonLauncher(s);
    return v.empty() ? std::string() : narrow(v[0]) + (v.size() > 1 ? " " + narrow(v[1]) : "");
}

std::wstring recon360QuoteArg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) return a;
    std::wstring out = L"\"";
    for (auto it = a.begin();; ++it) {
        size_t backslashes = 0;
        while (it != a.end() && *it == L'\\') { ++it; ++backslashes; }
        if (it == a.end()) { out.append(backslashes * 2, L'\\'); break; }
        if (*it == L'"') { out.append(backslashes * 2 + 1, L'\\'); out.push_back(L'"'); }
        else { out.append(backslashes, L'\\'); out.push_back(*it); }
    }
    out.push_back(L'"');
    return out;
}

std::string recon360QuoteArgUtf8(const std::string& a) {
    return narrow(recon360QuoteArg(widen(a)));
}

std::string recon360FindLichtfeld(const Recon360Settings& s) {
    if (!s.lichtfeldExe.empty() && isFile(widen(s.lichtfeldExe)))
        return s.lichtfeldExe;
    if (const wchar_t* env = _wgetenv(L"PF360_LICHTFELD")) {
        if (*env && isFile(env)) return narrow(env);
    }
    // Check beside our executable or in engines/lichtfeld beside exe
    const fs::path ed = exeDir();
    const fs::path beside = ed / L"LichtFeld-Studio.exe";
    if (isFile(beside.wstring())) return narrow(beside.wstring());
    const fs::path besideEngines = ed / L"engines" / L"lichtfeld" / L"LichtFeld-Studio.exe";
    if (isFile(besideEngines.wstring())) return narrow(besideEngines.wstring());

    // LocalAppData engine pack
    const fs::path enginePath = localAppData() / L"ViitorX" / L"PointForge" / L"engines" / L"lichtfeld" / L"LichtFeld-Studio.exe";
    if (isFile(enginePath.wstring())) return narrow(enginePath.wstring());
    wchar_t buf[MAX_PATH];
    if (SearchPathW(nullptr, L"LichtFeld-Studio.exe", nullptr, MAX_PATH, buf, nullptr))
        return narrow(buf);

    return {};
}

unsigned long recon360FindProcessByName(const std::wstring& exeName) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return 0;
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    unsigned long foundPid = 0;
    if (Process32FirstW(snap, &pe)) {
        do {
            if (_wcsicmp(pe.szExeFile, exeName.c_str()) == 0) {
                foundPid = pe.th32ProcessID;
                break;
            }
        } while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
    return foundPid;
}

bool recon360BringProcessWindowToFront(unsigned long pid) {
    if (!pid) return false;
    struct WndData { DWORD targetPid; HWND targetHwnd; } data{pid, nullptr};
    EnumWindows([](HWND hwnd, LPARAM lp) -> BOOL {
        auto* d = reinterpret_cast<WndData*>(lp);
        DWORD p = 0;
        GetWindowThreadProcessId(hwnd, &p);
        if (p == d->targetPid && IsWindowVisible(hwnd) && GetWindow(hwnd, GW_OWNER) == nullptr) {
            d->targetHwnd = hwnd;
            return FALSE;
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&data));
    if (data.targetHwnd) {
        ShowWindow(data.targetHwnd, SW_RESTORE);
        SetForegroundWindow(data.targetHwnd);
        return true;
    }
    return false;
}

LichtfeldEditorLaunchResult recon360LaunchLichtfeldEditor(const std::string& plyPath, const Recon360Settings& s) {
    LichtfeldEditorLaunchResult res;
    if (plyPath.empty()) {
        res.error = "No PLY file specified.";
        return res;
    }
    std::string exe = recon360FindLichtfeld(s);
    if (exe.empty()) {
        res.error = "LichtFeld-Studio.exe not found.";
        return res;
    }
    res.exePath = exe;

    fs::path exeFs = fs::u8path(exe);
    fs::path plyFs = fs::u8path(plyPath);
    if (!fs::exists(plyFs)) {
        res.error = "PLY file does not exist: " + plyPath;
        return res;
    }

    std::wstring wExe = exeFs.wstring();
    std::wstring wPly = plyFs.wstring();
    fs::path workDir = exeFs.parent_path();

    // Command line: "<exe>" -v "<ply>"
    std::wstring cmd = recon360QuoteArg(wExe) + L" -v " + recon360QuoteArg(wPly);
    std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back(L'\0');

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    // Launch detached without inheriting handles or sharing console
    const BOOL ok = CreateProcessW(wExe.c_str(), cmdBuf.data(), nullptr, nullptr, FALSE,
                                   DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
                                   nullptr, workDir.wstring().c_str(), &si, &pi);
    if (!ok) {
        res.error = "CreateProcessW failed (error " + std::to_string(GetLastError()) + ")";
        return res;
    }

    res.ok = true;
    res.pid = pi.dwProcessId;
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    logInfo("[LFS] Launched LichtFeld-Studio (PID " + std::to_string(res.pid) + ") for: " + plyPath);
    return res;
}

std::vector<Recon360Preset> recon360LoadPresets(std::string& err) {
    std::vector<Recon360Preset> out;
    const std::string root = recon360WorkerRoot();
    if (root.empty()) { err = "pf360 worker not found"; return out; }
    try {
        std::ifstream f(fs::path(widen(root)) / L"pf360" / L"presets.json");
        json j = json::parse(f);
        for (const char* key : {"preview", "balanced", "high"}) {
            const json& p = j.at("presets").at(key);
            Recon360Preset r;
            r.key = key;
            r.label = str(p, "label");
            r.description = str(p, "description");
            r.method = str(p, "method");
            r.fallbackMethod = str(p, "fallbackMethod");
            r.matchingMethod = str(p, "matchingMethod");
            r.costNote = str(p, "costNote");
            r.framesPerSecond = num(p, "framesPerSecond", 3.0);
            r.minFrames = (int)num(p, "minFrames", 30);
            r.maxFrames = (int)num(p, "maxFrames", 300);
            r.imagesPerEquirect = (int)num(p, "imagesPerEquirect", 8);
            r.downscaleFactor = (int)num(p, "downscaleFactor", 2);
            r.iterations = (int)num(p, "iterations", 30000);
            r.shDegree = (int)num(p, "shDegree", 3);
            r.useScaleRegularization = boolean(p, "useScaleRegularization");
            out.push_back(r);
        }
    } catch (const std::exception& e) {
        err = std::string("presets.json: ") + e.what();
        out.clear();
    }
    return out;
}

int recon360FramesForWindow(const Recon360Preset& p, double window) {
    if (window <= 0) return p.minFrames;
    return (int)std::max<double>(p.minFrames, std::min<double>(p.maxFrames, std::round(p.framesPerSecond * window)));
}

Recon360Doctor recon360RunDoctor(const Recon360Settings& s, bool quick) {
    Recon360Doctor d;
    std::string out, err;
    std::vector<std::wstring> args = {L"doctor", L"--json"};
    if (quick) args.push_back(L"--quick");
    if (!runWorkerCapture(s, args, quick ? 60000 : 300000, out, err)) { d.error = err; return d; }
    try {
        json j = json::parse(out);
        d.ran = true;
        d.ready = boolean(j, "ready");
        for (const auto& c : j.value("checks", json::array()))
            d.checks.push_back({str(c, "name"), str(c, "status"), str(c, "detail"), str(c, "fix")});
        d.methods = strList(j, "methods");
        d.masking = boolean(j, "masking");
        d.glomap = boolean(j, "glomap");
        d.vocabTree = boolean(j, "vocab_tree");
        const json tools = j.value("tools", json::object());
        d.maskModelDir = str(tools, "maskModel");
        d.nsPython = str(tools, "nerfstudioPython");
        d.lichtfeldExe = str(tools, "lichtfeld");
        const json t = j.value("torch", json::object());
        d.cuda = boolean(t, "cuda");
        d.torchVersion = str(t, "torch");
        d.nerfstudioVersion = str(t, "nerfstudio");
        const json g = j.value("gpu", json::object());
        if (g.contains("gpus") && g["gpus"].is_array() && !g["gpus"].empty()) {
            const json& g0 = g["gpus"][0];
            d.gpuName = str(g0, "name");
            d.gpuDriver = str(g0, "driver");
            // Python environment.py emits compute_cap as a string (e.g. "8.6")
            if (g0.contains("compute_cap")) {
                const json& cc = g0["compute_cap"];
                if (cc.is_number())
                    d.gpuComputeCap = (float)cc.get<double>();
                else if (cc.is_string()) {
                    try { d.gpuComputeCap = std::stof(cc.get<std::string>()); } catch (...) {}
                }
            }
            d.gpuTotalMB = (int)num(g0, "totalMB");
            d.gpuFreeMB = (int)num(g0, "freeMB");
        }
    } catch (const std::exception& e) {
        d.ran = false;
        d.error = std::string("could not parse doctor output: ") + e.what();
    }
    return d;
}

Recon360Probe recon360ProbeVideo(const Recon360Settings& s, const std::string& video, double start, double end) {
    Recon360Probe p;
    std::string out, err;
    std::vector<std::wstring> args = {L"probe", widen(video), L"--json", L"--quality"};
    char b[64];
    if (start > 0) { snprintf(b, sizeof(b), "%.3f", start); args.push_back(L"--start"); args.push_back(widen(b)); }
    if (end > 0)   { snprintf(b, sizeof(b), "%.3f", end);   args.push_back(L"--end");   args.push_back(widen(b)); }
    if (!runWorkerCapture(s, args, 300000, out, err)) { p.error = err; return p; }
    try {
        json j = json::parse(out);
        p.errors = strList(j, "errors");
        p.warnings = strList(j, "warnings");
        p.ok = boolean(j, "ok");
        p.fileName = str(j, "fileName");
        p.codec = str(j, "codec");
        p.pixelFormat = str(j, "pixelFormat");
        p.width = (int)num(j, "width");
        p.height = (int)num(j, "height");
        p.fps = num(j, "fps");
        p.durationSeconds = num(j, "durationSeconds");
        p.frameCount = (long long)num(j, "frameCount", -1);
        p.fileSizeBytes = (uint64_t)num(j, "fileSizeBytes");
        p.hasAudio = boolean(j, "hasAudio");
        if (j.contains("classification")) {
            const json& c = j["classification"];
            p.projection = str(c, "projection");
            p.projectionConfidence = str(c, "confidence");
            p.supported = boolean(c, "supported");
            p.requiresConfirmation = boolean(c, "requiresConfirmation");
            p.reasons = strList(c, "reasons");
        }
        if (!p.ok && p.error.empty() && !p.errors.empty()) p.error = p.errors.front();
    } catch (const std::exception& e) {
        p.ok = false;
        p.error = std::string("could not parse probe output: ") + e.what();
    }
    return p;
}

bool recon360ExtractFrame(const Recon360Settings& s, const std::string& video, double t, int width,
                          std::vector<unsigned char>& rgb, int& outW, int& outH, std::string& err) {
    const fs::path tmp = fs::temp_directory_path() / widen("pf360_preview_" + std::to_string(GetCurrentProcessId()) + ".rgb");
    char tb[32];
    snprintf(tb, sizeof(tb), "%.3f", t);
    std::string out;
    if (!runWorkerCapture(s, {L"frame", widen(video), L"--time", widen(tb), L"--width", std::to_wstring(width),
                              L"--out", tmp.wstring()}, 120000, out, err))
        return false;
    try {
        json j = json::parse(out);
        if (!boolean(j, "ok")) { err = str(j, "error"); return false; }
        outW = (int)num(j, "width");
        outH = (int)num(j, "height");
        std::ifstream f(tmp, std::ios::binary);
        rgb.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        f.close();
        std::error_code ec;
        fs::remove(tmp, ec);
        if ((int)rgb.size() != outW * outH * 3) { err = "preview frame has the wrong size"; return false; }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool recon360MaskPreview(const Recon360Settings& s, const std::vector<unsigned char>& rgb, int w, int h,
                         const std::string& groups, int dilatePx, std::vector<unsigned char>& overlayRgb,
                         float& maskedFraction, std::string& err) {
    if (w <= 0 || h <= 0 || (int)rgb.size() != w * h * 3) { err = "no preview frame"; return false; }
    const std::string tag = std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64());
    const fs::path in = fs::temp_directory_path() / widen("pf360_mask_in_" + tag + ".rgb");
    const fs::path out = fs::temp_directory_path() / widen("pf360_mask_out_" + tag + ".rgb");
    std::error_code ec;
    {
        std::ofstream f(in, std::ios::binary | std::ios::trunc);
        if (!f) { err = "cannot write temp image"; return false; }
        f.write((const char*)rgb.data(), (std::streamsize)rgb.size());
    }
    std::string res;
    const bool ran = runWorkerCapture(s, {L"mask-preview", L"--in", in.wstring(), L"--size", std::to_wstring(w),
                                          std::to_wstring(h), L"--out", out.wstring(), L"--groups", widen(groups),
                                          L"--dilate", std::to_wstring(dilatePx)},
                                      600000, res, err);
    fs::remove(in, ec);
    if (!ran) { fs::remove(out, ec); return false; }
    try {
        json j = json::parse(res);
        if (!boolean(j, "ok")) { err = str(j, "error"); fs::remove(out, ec); return false; }
        maskedFraction = (float)num(j, "maskedFraction");
        std::ifstream f(out, std::ios::binary);
        overlayRgb.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        f.close();
        fs::remove(out, ec);
        if ((int)overlayRgb.size() != w * h * 3) { err = "mask preview has the wrong size"; return false; }
        return true;
    } catch (const std::exception& e) {
        fs::remove(out, ec);
        err = e.what();
        return false;
    }
}

bool recon360WriteManifest(const Recon360JobSpec& sp, const Recon360Settings& s, std::string& path,
                           std::string& err) {
    json m;
    m["schemaVersion"] = 1;
    m["jobId"] = sp.jobId;
    m["jobType"] = "360_video_to_gaussian_splat";
    m["input"] = {{"videoPath", sp.videoPath}, {"projection", "equirectangular"},
                  {"projectionConfirmed", sp.projectionConfirmed}, {"startTimeSeconds", sp.startSeconds},
                  {"endTimeSeconds", sp.endSeconds > 0 ? json(sp.endSeconds) : json(nullptr)}};
    m["processing"] = {{"targetFrames", sp.targetFrames}, {"imagesPerEquirect", sp.imagesPerEquirect},
                       {"cropTop", sp.cropTop}, {"cropBottom", sp.cropBottom}, {"cropLeft", sp.cropLeft},
                       {"cropRight", sp.cropRight}, {"matchingMethod", sp.matchingMethod},
                       {"cameraSolver", sp.cameraSolver == "glomap" ? "glomap" : "colmap"},
                       {"loopDetection", sp.loopDetection && sp.matchingMethod == "sequential"},
                       {"globalRefinement", sp.globalRefinement},
                       {"gpuBundleAdjustment", sp.gpuBundleAdjustment}};
    {
        json groups = json::array();
        if (sp.maskPeople)   groups.push_back("people");
        if (sp.maskVehicles) groups.push_back("vehicles");
        if (sp.maskAnimals)  groups.push_back("animals");
        if (sp.maskSky)      groups.push_back("sky");
        m["processing"]["masking"] = {{"enabled", sp.maskEnabled && !groups.empty()}, {"groups", groups},
                                      {"dilatePx", std::max(0, std::min(64, sp.maskDilatePx))}};
    }
    json tr = {{"engine", sp.trainingEngine.empty() ? "nerfstudio" : sp.trainingEngine},
               {"qualityPreset", sp.qualityPreset}, {"method", sp.method}, {"iterations", sp.iterations},
               {"downscaleFactor", sp.downscaleFactor}, {"shDegree", sp.shDegree}, {"randomSeed", sp.randomSeed},
               {"useScaleRegularization", sp.useScaleRegularization}};
    if (sp.trainingEngine == "lichtfeld") {
        tr["lfsStrategy"] = sp.lfsStrategy.empty() ? "mrnf" : sp.lfsStrategy;
        tr["maxGaussians"] = sp.maxGaussians > 0 ? sp.maxGaussians : 2000000;
    }
    if (!sp.fallbackMethod.empty()) tr["fallbackMethod"] = sp.fallbackMethod;
    if (sp.cullAlphaThresh > 0)   tr["cullAlphaThresh"] = sp.cullAlphaThresh;
    if (sp.densifyGradThresh > 0) tr["densifyGradThresh"] = sp.densifyGradThresh;
    if (sp.cullScaleThresh > 0)   tr["cullScaleThresh"] = sp.cullScaleThresh;
    m["training"] = tr;
    m["output"] = {{"directory", sp.outputDir}, {"format", "ply"}, {"keepIntermediate", true}};
    if (!sp.plyFileName.empty()) m["output"]["plyFileName"] = sp.plyFileName;
    json env = {{"backend", "native"}};
    if (!s.nsPython.empty())   env["nerfstudioPython"] = s.nsPython;
    if (!s.colmapExe.empty())  env["colmapExe"] = s.colmapExe;
    if (!s.ffmpegExe.empty())  env["ffmpegExe"] = s.ffmpegExe;
    if (!s.ffprobeExe.empty()) env["ffprobeExe"] = s.ffprobeExe;
    if (!s.maskModelDir.empty()) env["maskModelDir"] = s.maskModelDir;
    if (!s.lichtfeldExe.empty()) env["lichtfeldExe"] = s.lichtfeldExe;
    m["environment"] = env;
    json opts = {{"allowCache", sp.allowCache}};
    if (!sp.stopAfter.empty()) opts["stopAfter"] = sp.stopAfter;
    m["options"] = opts;

    std::error_code ec;
    const fs::path dir = localAppData() / L"ViitorX" / L"PointForge" / L"pf360-jobs";
    fs::create_directories(dir, ec);
    const fs::path p = dir / widen(sp.jobId + ".json");
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) { err = "cannot write manifest " + narrow(p.wstring()); return false; }
    f << m.dump(2);
    path = narrow(p.wstring());
    return (bool)f;
}

bool recon360SpecFromManifest(const std::string& text, Recon360JobSpec& sp, std::string& err) {
    try {
        const json m = json::parse(text);
        sp = Recon360JobSpec{};
        sp.jobId = m.at("jobId").get<std::string>();
        const json& in = m.at("input");
        sp.videoPath = in.at("videoPath").get<std::string>();
        sp.projectionConfirmed = in.value("projectionConfirmed", false);
        sp.startSeconds = in.value("startTimeSeconds", 0.0);
        sp.endSeconds = in.contains("endTimeSeconds") && in["endTimeSeconds"].is_number() ? in["endTimeSeconds"].get<double>() : -1;
        const json& pr = m.at("processing");
        sp.targetFrames = pr.value("targetFrames", 90);
        sp.imagesPerEquirect = pr.value("imagesPerEquirect", 8);
        sp.cropTop = pr.value("cropTop", 0.0f); sp.cropBottom = pr.value("cropBottom", 0.0f);
        sp.cropLeft = pr.value("cropLeft", 0.0f); sp.cropRight = pr.value("cropRight", 0.0f);
        sp.matchingMethod = pr.value("matchingMethod", "sequential");
        sp.cameraSolver = pr.value("cameraSolver", "colmap");
        sp.loopDetection = pr.value("loopDetection", false);
        sp.globalRefinement = pr.value("globalRefinement", "balanced");
        sp.gpuBundleAdjustment = pr.value("gpuBundleAdjustment", false);
        if (pr.contains("masking") && pr["masking"].is_object()) {
            const json& mk = pr["masking"];
            sp.maskEnabled = mk.value("enabled", false);
            sp.maskDilatePx = mk.value("dilatePx", 8);
            sp.maskPeople = sp.maskVehicles = sp.maskAnimals = sp.maskSky = false;
            for (const auto& g : mk.value("groups", json::array())) {
                const std::string s = g.get<std::string>();
                sp.maskPeople |= s == "people"; sp.maskVehicles |= s == "vehicles";
                sp.maskAnimals |= s == "animals"; sp.maskSky |= s == "sky";
            }
        }
        const json& tr = m.at("training");
        sp.trainingEngine = tr.value("engine", "nerfstudio");
        sp.lfsStrategy = tr.value("lfsStrategy", "mrnf");
        sp.maxGaussians = tr.value("maxGaussians", 2000000);
        sp.qualityPreset = tr.value("qualityPreset", "balanced");
        sp.method = tr.value("method", "splatfacto");
        sp.fallbackMethod = tr.value("fallbackMethod", "");
        sp.iterations = tr.value("iterations", 30000);
        sp.downscaleFactor = tr.value("downscaleFactor", 2);
        sp.shDegree = tr.value("shDegree", 3);
        sp.randomSeed = tr.value("randomSeed", 42);
        sp.useScaleRegularization = tr.value("useScaleRegularization", false);
        sp.cullAlphaThresh = tr.value("cullAlphaThresh", -1.0f);
        sp.densifyGradThresh = tr.value("densifyGradThresh", -1.0f);
        sp.cullScaleThresh = tr.value("cullScaleThresh", -1.0f);
        const json& out = m.at("output");
        sp.outputDir = out.at("directory").get<std::string>();
        sp.plyFileName = out.value("plyFileName", "");
        const json opts = m.value("options", json::object());
        sp.allowCache = opts.value("allowCache", true);
        sp.stopAfter = opts.value("stopAfter", "");
    } catch (const std::exception& e) {
        err = std::string("manifest: ") + e.what();
        return false;
    }
    return true;
}

std::vector<Recon360Unfinished> recon360ListUnfinished() {
    std::vector<Recon360Unfinished> out;
    std::error_code ec;
    const fs::path dir = localAppData() / L"ViitorX" / L"PointForge" / L"pf360-jobs";
    if (!fs::is_directory(dir, ec)) return out;
    struct Cand { fs::path manifest; fs::file_time_type t; };
    std::vector<Cand> cands;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file(ec) && e.path().extension() == L".json") cands.push_back({e.path(), e.last_write_time(ec)});
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.t > b.t; });
    std::vector<std::string> seen;   // one entry per workspace (retries reuse it)
    for (const auto& c : cands) {
        std::ifstream f(c.manifest, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        Recon360Unfinished u;
        std::string err;
        if (!recon360SpecFromManifest(text, u.spec, err)) continue;
        const fs::path ws = fs::u8path(u.spec.outputDir);
        std::string key = ws.lexically_normal().u8string();
        for (char& ch : key) ch = (char)std::tolower((unsigned char)ch);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
        seen.push_back(key);
        json st;
        try {
            std::ifstream sf(ws / "state.json", std::ios::binary);
            if (!sf) continue;                                     // never started or deleted
            sf >> st;
        } catch (...) { continue; }
        const json stages = st.value("stages", json::object());
        // Finished: validation done (or the requested stopAfter stage done).
        const std::string endStage = u.spec.stopAfter.empty() ? "validation" : u.spec.stopAfter;
        if (stages.contains(endStage) && stages[endStage].value("status", "") == "completed") continue;
        for (int i = 0; i < Recon360Live::kStageCount; ++i) {
            const char* id = Recon360Live::kStages[i];
            if (!stages.contains(id)) continue;
            const std::string s = stages[id].value("status", "");
            u.updatedAt = std::max(u.updatedAt, stages[id].value("finishedAt", 0.0));
            if (s != "completed") { u.lastStage = id; u.lastStatus = s; }
        }
        u.manifestPath = narrow(c.manifest.wstring());
        out.push_back(std::move(u));
        if (out.size() >= 100) break;
    }
    return out;
}

bool recon360Run(const Recon360JobSpec& spec, const Recon360Settings& s, std::atomic<bool>& cancel,
                 Recon360Live& live, const std::function<void(float, const std::string&)>& progress,
                 std::string& err) {
    std::string manifest;
    if (!recon360WriteManifest(spec, s, manifest, err)) return false;
    const std::string root = recon360WorkerRoot();
    if (root.empty()) { err = "pf360 worker files not found"; return false; }
    std::vector<std::wstring> argv = pythonLauncher(s);
    if (argv.empty()) { err = "No Python 3 interpreter found (Preferences > 360 Reconstruction)"; return false; }
    for (const wchar_t* a : {L"-m", L"pf360", L"run", L"--stdin-control", L"--manifest"}) argv.push_back(a);
    argv.push_back(widen(manifest));

    Child c;
    if (!spawn(argv, widen(root), buildEnv(s, widen(root)), true, c, err)) return false;
    logInfo("[pf360] job " + spec.jobId + " started (manifest " + manifest + ")");

    // stderr = human diagnostics (tracebacks, tool noise) -> Console + live log
    std::thread te = lineReader(c.errRd, [&](const std::string& l) {
        logDebug("[pf360] " + l);
        std::lock_guard<std::mutex> lk(live.mu);
        live.addLog(l);
    });

    bool completed = false, failed = false, cancelled = false;
    std::string failMsg;
    std::thread to = lineReader(c.outRd, [&](const std::string& line) {
        json ev;
        try { ev = json::parse(line); } catch (...) {
            std::lock_guard<std::mutex> lk(live.mu);
            live.addLog(line);                      // tolerate stray non-JSON output
            return;
        }
        if (num(ev, "v", 0) != 1) return;           // unknown protocol version: ignore
        const std::string type = str(ev, "type"), stage = str(ev, "stage"), msg = str(ev, "message");
        int si = -1;
        for (int i = 0; i < Recon360Live::kStageCount; ++i) if (stage == Recon360Live::kStages[i]) si = i;
        float overall = -2;
        std::string status;
        {
            std::lock_guard<std::mutex> lk(live.mu);
            if (type == "stage_started" && si >= 0) {
                live.stage[si] = Recon360Live::StageState::Running;
                live.currentStage = si;
                live.stagePercent = -1;
                live.stageMessage.clear();
                live.addLog(std::string("> ") + recon360StageLabel(si));
                overall = recon360Overall(si, 0);
                status = recon360StageLabel(si);
            } else if (type == "progress" && si >= 0) {
                const auto it = ev.find("percent");
                live.stagePercent = (it != ev.end() && it->is_number()) ? it->get<float>() : -1.0f;
                live.stageMessage = msg;
                overall = live.stagePercent >= 0 ? recon360Overall(si, live.stagePercent) : -1.0f;
                status = std::string(recon360StageLabel(si)) + (msg.empty() ? "" : ": " + msg);
            } else if ((type == "stage_completed" || type == "stage_skipped") && si >= 0) {
                // stage_skipped carries data.enabled == false for optional stages that are off.
                const bool off = type == "stage_skipped" && ev.contains("data") && ev["data"].is_object() &&
                                 ev["data"].value("enabled", true) == false;
                live.stage[si] = type == "stage_completed" ? Recon360Live::StageState::Done
                                 : off ? Recon360Live::StageState::Off : Recon360Live::StageState::Skipped;
                live.addLog(std::string(recon360StageLabel(si)) +
                            (type == "stage_completed" ? " - done" : off ? " - off" : " - reused cached result"));
                overall = recon360Overall(si + 1, 0);
            } else if (type == "log") {
                live.addLog(msg);
            } else if (type == "warning") {
                live.warnings.push_back(msg);
                live.addLog("WARNING: " + msg);
            } else if (type == "error") {
                if (si >= 0) live.stage[si] = Recon360Live::StageState::Failed;
                live.addLog("ERROR: " + msg);
            } else if (type == "cancellation_requested") {
                live.addLog("Cancellation requested...");
            } else if (type == "job_completed" || type == "job_failed" || type == "job_cancelled") {
                live.finalMessage = msg;
                if (ev.contains("data")) {
                    const json& d = ev["data"];
                    live.reportJson = d.dump(2);
                    live.workspace = str(d, "workspace");
                    if (d.contains("output")) {
                        live.gaussianCount = (uint64_t)num(d["output"], "gaussianCount");
                        live.plyBytes = (uint64_t)num(d["output"], "fileSizeBytes");
                    }
                }
                if (type == "job_completed") { completed = true; live.outputPly = str(ev, "outputPath"); }
                if (type == "job_failed") { failed = true; failMsg = msg; }
                if (type == "job_cancelled") cancelled = true;
                live.addLog(msg);
            }
        }
        if (overall > -2 && progress) progress(overall, status.empty() ? msg : status);
        if (type == "warning") logWarn("[pf360] " + msg);
        if (type == "job_failed") logError("[pf360] " + msg);
    });

    // Poll for cancellation while the worker runs.
    bool cancelSent = false;
    auto cancelAt = std::chrono::steady_clock::time_point{};
    for (;;) {
        if (WaitForSingleObject(c.process, 150) == WAIT_OBJECT_0) break;
        if (cancel.load() && !cancelSent) {
            cancelSent = true;
            cancelAt = std::chrono::steady_clock::now();
            const char msg[] = "cancel\n";
            DWORD wr = 0;
            WriteFile(c.inWr, msg, sizeof(msg) - 1, &wr, nullptr);
            logInfo("[pf360] cancel sent to worker");
        }
        // Grace period for the worker to save a restore point (training writes a checkpoint, up to
        // ~150 s for very large models), stop its children and write its report. On app exit
        // there is no waiting: the last periodic checkpoint is the restore point.
        const auto grace = g_recon360ShuttingDown.load() ? std::chrono::seconds(0) : std::chrono::seconds(180);
        if (cancelSent && std::chrono::steady_clock::now() - cancelAt > grace) {
            logWarn("[pf360] worker did not stop in time; terminating its process tree");
            c.terminateTree();
            WaitForSingleObject(c.process, 5000);
            break;
        }
    }
    // The worker exited: anything it left behind in the job dies now.
    c.terminateTree();
    to.join();
    te.join();
    DWORD code = 1;
    GetExitCodeProcess(c.process, &code);
    logInfo("[pf360] job " + spec.jobId + " exited with code " + std::to_string(code));

    if (cancel.load() || cancelled) { err = "Canceled"; return false; }
    if (failed) { err = failMsg; return false; }
    if (!completed) { err = "Worker exited unexpectedly (code " + std::to_string(code) + ") - see Console"; return false; }
    if (!spec.stopAfter.empty() && live.outputPly.empty()) return true;   // partial run requested
    // Host-side compatibility check with the existing reader (header sniff).
    if (live.outputPly.empty() ||
        SplatReader::detectFormat(live.outputPly) != SplatFileFormat::PlyGaussians) {
        err = "Worker reported success but the output is not a Gaussian-splat PLY readable by PointForge";
        return false;
    }
    return true;
}

bool recon360IsZUpOutput(const std::string& plyPath) {
    std::error_code ec;
    const fs::path sidecar = fs::path(widen(plyPath + ".pf360.json"));
    if (fs::exists(sidecar, ec)) {
        try {
            std::ifstream f(sidecar);
            json j = json::parse(f);
            if (j.value("upAxis", "") == "z_up") return true;
        } catch (...) {}
    }
    const fs::path ply = fs::path(widen(plyPath));
    const fs::path report = ply.parent_path().parent_path() / L"reconstruction_report.json";
    if (!fs::exists(report, ec)) return false;
    try {
        std::ifstream f(report);
        json j = json::parse(f);
        const json& o = j.at("output");
        if (str(o, "upAxis") != "z_up") return false;
        return fs::equivalent(fs::path(widen(str(o, "path"))), ply, ec);
    } catch (...) {
        return false;
    }
}

Recon360CameraPath recon360CameraPath(const std::string& plyPath) {
    Recon360CameraPath out;
    std::error_code ec;
    const fs::path ply = fs::path(widen(plyPath));
    const fs::path ws = ply.parent_path().parent_path();
    fs::path tfPath = ws / L"processed" / L"transforms.json";

    // Check sidecar or report to determine coordinate frame and optional levelled camerasFile
    std::string frameType = "ns_dataparser";
    const fs::path sidecarPath = fs::path(widen(plyPath + ".pf360.json"));
    if (fs::exists(sidecarPath, ec)) {
        try {
            std::ifstream sf(sidecarPath);
            json sc = json::parse(sf);
            frameType = sc.value("frame", "ns_dataparser");
            const std::string camFile = sc.value("camerasFile", std::string{});
            if (!camFile.empty()) {
                fs::path cp = fs::path(widen(camFile));
                if (cp.is_relative()) cp = ws / cp;
                if (fs::exists(cp, ec)) tfPath = cp;
            }
        } catch (...) {}
    } else if (fs::exists(ws / L"reconstruction_report.json", ec)) {
        try {
            std::ifstream rf(ws / L"reconstruction_report.json");
            json rj = json::parse(rf);
            if (rj.contains("training") && rj["training"].value("engine", "") == "lichtfeld") {
                frameType = "transforms";
            }
        } catch (...) {}
    }
    if (!fs::exists(tfPath, ec)) return out;

    if (frameType == "transforms") {
        try {
            std::ifstream ftf(tfPath);
            const json tf = json::parse(ftf);
            double upSum[3] = {0, 0, 0};
            for (const json& fr : tf.at("frames")) {
                const json& m = fr.at("transform_matrix");
                std::array<double, 3> P{}, F{};
                for (int r = 0; r < 3; ++r) {
                    P[r] = m.at(r).at(3).get<double>();
                    F[r] = -m.at(r).at(2).get<double>();   // OpenGL camera looks down -Z
                    upSum[r] += m.at(r).at(1).get<double>();
                }
                const double fl = std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
                if (fl > 0) for (double& v : F) v /= fl;
                out.pos.push_back(P);
                out.fwd.push_back(F);
            }
            const double ul = std::sqrt(upSum[0] * upSum[0] + upSum[1] * upSum[1] + upSum[2] * upSum[2]);
            if (out.pos.empty() || !(ul > 0)) return Recon360CameraPath{};
            for (int r = 0; r < 3; ++r) out.up[r] = upSum[r] / ul;
            out.ok = true;
            return out;
        } catch (...) {
            return Recon360CameraPath{};
        }
    }

    if (!fs::exists(ws / L"reconstruction_report.json", ec)) return out;
    // dataparser_transforms.json of the training run (newest if several).
    fs::path dpPath;
    fs::file_time_type dpTime{};
    for (fs::recursive_directory_iterator it(ws / L"training", fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it->path().filename() != L"dataparser_transforms.json") continue;
        const auto t = fs::last_write_time(it->path(), ec);
        if (dpPath.empty() || t > dpTime) { dpPath = it->path(); dpTime = t; }
    }
    if (dpPath.empty()) return out;
    try {
        std::ifstream fdp(dpPath), ftf(tfPath);
        const json dp = json::parse(fdp), tf = json::parse(ftf);
        double T[3][4];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) T[r][c] = dp.at("transform").at(r).at(c).get<double>();
        const double s = dp.at("scale").get<double>();
        // Nerfstudio folds transforms.json's "applied_transform" (A, a pure
        // rotation from ns-process-data) into the saved transform, so the frames
        // of transforms.json map by T * A^-1 = T * A^T. Without this the cameras
        // land rotated in the splat (verified against capture photos).
        if (tf.contains("applied_transform")) {
            double A[3][3], TA[3][3];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) A[r][c] = tf["applied_transform"].at(r).at(c).get<double>();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    TA[r][c] = T[r][0] * A[c][0] + T[r][1] * A[c][1] + T[r][2] * A[c][2];   // T * A^T
            // Translation: A's own translation column is zero for ns-process-data.
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) T[r][c] = TA[r][c];
        }
        double upSum[3] = {0, 0, 0};
        for (const json& fr : tf.at("frames")) {
            const json& m = fr.at("transform_matrix");
            double p[3], f[3], u[3];
            for (int r = 0; r < 3; ++r) {
                p[r] = m.at(r).at(3).get<double>();
                f[r] = -m.at(r).at(2).get<double>();   // OpenGL camera looks down -Z
                u[r] = m.at(r).at(1).get<double>();
            }
            std::array<double, 3> P{}, F{};
            for (int r = 0; r < 3; ++r) {
                P[r] = s * (T[r][0] * p[0] + T[r][1] * p[1] + T[r][2] * p[2] + T[r][3]);
                F[r] = T[r][0] * f[0] + T[r][1] * f[1] + T[r][2] * f[2];
                upSum[r] += T[r][0] * u[0] + T[r][1] * u[1] + T[r][2] * u[2];
            }
            const double fl = std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
            if (fl > 0) for (double& v : F) v /= fl;
            out.pos.push_back(P);
            out.fwd.push_back(F);
        }
        const double ul = std::sqrt(upSum[0] * upSum[0] + upSum[1] * upSum[1] + upSum[2] * upSum[2]);
        if (out.pos.empty() || !(ul > 0)) return Recon360CameraPath{};
        for (int r = 0; r < 3; ++r) out.up[r] = upSum[r] / ul;
        out.ok = true;
    } catch (...) {
        return Recon360CameraPath{};
    }
    return out;
}

void recon360Reveal(const std::string& path, bool selectFile) {
    std::wstring w = widen(path);
    if (selectFile) {
        std::wstring args = L"/select,\"" + w + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
    } else {
        ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
}

#else  // ---------------------------------------------------------------- stubs
std::string recon360WorkerRoot() { return {}; }
std::string recon360FindPython(const Recon360Settings&) { return {}; }
std::string recon360FindLichtfeld(const Recon360Settings&) { return {}; }
std::vector<Recon360Preset> recon360LoadPresets(std::string& err) { err = "Windows only"; return {}; }
int recon360FramesForWindow(const Recon360Preset& p, double) { return p.minFrames; }
Recon360Doctor recon360RunDoctor(const Recon360Settings&, bool) { Recon360Doctor d; d.error = "Windows only"; return d; }
Recon360Probe recon360ProbeVideo(const Recon360Settings&, const std::string&, double, double) {
    Recon360Probe p; p.error = "Windows only"; return p;
}
bool recon360ExtractFrame(const Recon360Settings&, const std::string&, double, int, std::vector<unsigned char>&,
                          int&, int&, std::string& err) { err = "Windows only"; return false; }
bool recon360MaskPreview(const Recon360Settings&, const std::vector<unsigned char>&, int, int, const std::string&,
                         int, std::vector<unsigned char>&, float&, std::string& err) {
    err = "Windows only"; return false;
}
bool recon360WriteManifest(const Recon360JobSpec&, const Recon360Settings&, std::string&, std::string& err) {
    err = "Windows only"; return false;
}
bool recon360Run(const Recon360JobSpec&, const Recon360Settings&, std::atomic<bool>&, Recon360Live&,
                 const std::function<void(float, const std::string&)>&, std::string& err) {
    err = "360 reconstruction is only implemented on Windows"; return false;
}
bool recon360IsZUpOutput(const std::string&) { return false; }
Recon360CameraPath recon360CameraPath(const std::string&) { return {}; }
void recon360Reveal(const std::string&, bool) {}
std::vector<Recon360Unfinished> recon360ListUnfinished() { return {}; }
bool recon360SpecFromManifest(const std::string&, Recon360JobSpec&, std::string& err) { err = "Windows only"; return false; }
std::wstring recon360QuoteArg(const std::wstring& a) { return a; }
std::string recon360QuoteArgUtf8(const std::string& a) { return a; }
unsigned long recon360FindProcessByName(const std::wstring&) { return 0; }
bool recon360BringProcessWindowToFront(unsigned long) { return false; }
LichtfeldEditorLaunchResult recon360LaunchLichtfeldEditor(const std::string&, const Recon360Settings&) {
    LichtfeldEditorLaunchResult r; r.error = "Windows only"; return r;
}
#endif

} // namespace pf
