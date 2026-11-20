#include "EnginePacks.h"
#include "EnginePackKey.h"
#include "common/Log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
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
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

#ifndef PF_ENGINE_CATALOG_URL
#define PF_ENGINE_CATALOG_URL ""
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace pf {

std::string engineDefaultCatalogUrl() { return PF_ENGINE_CATALOG_URL; }
bool engineHasSigningKey() { return std::string(kEnginePackPublicKeyHex).size() == 128; }

// ---------------------------------------------------------------------------
// validation (everything in the catalog is untrusted until the signature checks,
// and is still validated after: a signed catalog must not be able to write
// outside the engines folder or run arbitrary commands)
// ---------------------------------------------------------------------------
namespace {

bool validId(const std::string& s) { return std::regex_match(s, std::regex("[a-z0-9][a-z0-9._-]{0,63}")); }

bool validName(const std::string& s) {
    return s.size() <= 128 && s != "." && s != ".." && std::regex_match(s, std::regex("[A-Za-z0-9._-]+"));
}

bool validRelDir(const std::string& s) {
    if (s.empty() || s.size() > 200) return false;
    std::stringstream ss(s);
    std::string part;
    int n = 0;
    while (std::getline(ss, part, '/')) {
        if (!validName(part)) return false;
        if (++n > 4) return false;
    }
    return s.back() != '/';
}

bool isLoopbackHttp(const std::string& u) {
    return u.rfind("http://127.0.0.1:", 0) == 0 || u.rfind("http://localhost:", 0) == 0;
}

bool validUrl(const std::string& u) {
    if (u.size() > 2000) return false;
    if (u.rfind("https://", 0) != 0 && !isLoopbackHttp(u)) return false;
    for (unsigned char c : u)
        if (c <= 0x20 || c == '"' || c == '\'' || c == '<' || c == '>' || c == '\\' || c >= 0x7f) return false;
    return true;
}

bool validSha(const std::string& s) { return std::regex_match(s, std::regex("[0-9a-f]{64}")); }

std::string fmtGB(uint64_t b) {
    char buf[32];
    if (b >= (1ull << 30)) snprintf(buf, sizeof(buf), "%.2f GB", (double)b / (1ull << 30));
    else snprintf(buf, sizeof(buf), "%.0f MB", (double)b / (1ull << 20));
    return buf;
}

std::string readFile(const fs::path& p, size_t maxBytes) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (s.size() > maxBytes) s.clear();
    return s;
}

bool writeFileAtomic(const fs::path& p, const std::string& data) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.wstring() + L".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(data.data(), (std::streamsize)data.size());
        if (!f) return false;
    }
    fs::rename(tmp, p, ec);
    return !ec;
}

uint64_t fileSize(const fs::path& p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : (uint64_t)n;
}

std::string base64Decode(const std::string& in) {
    static const std::string tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    unsigned val = 0;
    int bits = -8;
    for (char c : in) {
        if (c == '=' || c == '\r' || c == '\n' || c == ' ') continue;
        const size_t pos = tbl.find(c);
        if (pos == std::string::npos) return {};
        val = (val << 6) | (unsigned)pos;
        bits += 6;
        if (bits >= 0) {
            out.push_back((char)((val >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

bool hexDecode(const std::string& hex, std::vector<unsigned char>& out) {
    if (hex.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < hex.size(); i += 2) {
        unsigned v = 0;
        if (sscanf(hex.c_str() + i, "%2x", &v) != 1) return false;
        out.push_back((unsigned char)v);
    }
    return true;
}

std::string nowStamp(const char* fmt) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[64];
    std::strftime(b, sizeof(b), fmt, &tm);
    return b;
}

} // namespace

// ---------------------------------------------------------------------------
// catalog parsing
// ---------------------------------------------------------------------------
bool engineParseCatalog(const std::string& text, EngineCatalog& out, std::string& err) {
    out = EngineCatalog{};
    try {
        const json j = json::parse(text);
        if (j.value("schema", 0) != 1) { err = "unsupported catalog schema (update PointForge)"; return false; }
        out.sequence = j.at("sequence").get<int64_t>();
        out.created = j.value("created", "");
        for (const auto& pj : j.at("packs")) {
            EnginePack p;
            p.id = pj.at("id").get<std::string>();
            p.title = pj.value("title", p.id);
            p.description = pj.value("description", "");
            p.version = pj.at("version").get<std::string>();
            p.group = pj.value("group", "");
            p.installDir = pj.at("installDir").get<std::string>();
            p.postInstall = pj.value("postInstall", "");
            p.license = pj.value("license", "");
            p.required = pj.value("required", false);
            p.installedBytes = pj.value("installedBytes", (uint64_t)0);
            if (!validId(p.id)) { err = "invalid pack id"; return false; }
            if (!validRelDir(p.installDir)) { err = "invalid installDir for " + p.id; return false; }
            if (p.postInstall != "" && p.postInstall != "python-relocate") {
                err = "unknown postInstall step for " + p.id;
                return false;
            }
            if (p.version.empty() || p.version.size() > 120) { err = "invalid version for " + p.id; return false; }
            for (const auto& fj : pj.at("files")) {
                EnginePackFile f;
                f.name = fj.at("name").get<std::string>();
                f.kind = fj.at("kind").get<std::string>();
                f.url = fj.at("url").get<std::string>();
                f.sha256 = fj.at("sha256").get<std::string>();
                f.size = fj.at("size").get<uint64_t>();
                f.fileCount = fj.value("fileCount", (uint64_t)0);
                if (!validName(f.name) || (f.kind != "zip" && f.kind != "file") || !validUrl(f.url) ||
                    !validSha(f.sha256) || f.size == 0 || f.size > (64ull << 30)) {
                    err = "invalid file entry in pack " + p.id;
                    return false;
                }
                p.files.push_back(f);
            }
            if (p.files.empty()) { err = "pack " + p.id + " lists no files"; return false; }
            if (out.find(p.id)) { err = "duplicate pack " + p.id; return false; }
            out.packs.push_back(std::move(p));
        }
    } catch (const std::exception& e) {
        err = std::string("catalog is not valid JSON: ") + e.what();
        return false;
    }
    return true;
}

EnginePackStatus engineStatus(const std::string& root, const EnginePack& pack) {
    EnginePackStatus st;
    const fs::path dir = fs::u8path(root) / fs::u8path(pack.installDir);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return st;
    const std::string s = readFile(dir / ".pfpack.json", 1 << 16);
    if (s.empty()) { st.state = EnginePackStatus::State::SelfManaged; return st; }
    try {
        const json j = json::parse(s);
        if (j.value("id", "") != pack.id) { st.state = EnginePackStatus::State::SelfManaged; return st; }
        st.installedVersion = j.value("version", "");
        st.state = st.installedVersion == pack.version ? EnginePackStatus::State::Installed
                                                       : EnginePackStatus::State::UpdateAvailable;
    } catch (...) {
        st.state = EnginePackStatus::State::SelfManaged;
    }
    return st;
}

uint64_t engineRequiredBytes(const EngineCatalog& cat, const std::vector<std::string>& ids) {
    uint64_t n = 0;
    for (const auto& id : ids)
        if (const EnginePack* p = cat.find(id)) n += p->downloadBytes() + p->installedBytes;
    return n + n / 10 + (512ull << 20);   // 10 % + 512 MB margin
}

#ifdef _WIN32
// ---------------------------------------------------------------------------
// Windows plumbing
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
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) return a;
    std::wstring out = L"\"";
    for (auto it = a.begin();; ++it) {
        size_t bs = 0;
        while (it != a.end() && *it == L'\\') { ++it; ++bs; }
        if (it == a.end()) { out.append(bs * 2, L'\\'); break; }
        if (*it == L'"') { out.append(bs * 2 + 1, L'\\'); out.push_back(L'"'); }
        else { out.append(bs, L'\\'); out.push_back(*it); }
    }
    out.push_back(L'"');
    return out;
}

// curl.exe / tar.exe from System32 by absolute path (never whatever is on PATH).
std::wstring systemTool(const wchar_t* exe) {
    wchar_t dir[MAX_PATH];
    const UINT n = GetSystemDirectoryW(dir, MAX_PATH);
    return n ? std::wstring(dir, n) + L"\\" + exe : std::wstring(exe);
}

constexpr int kSpawnFailed = -1000;
constexpr int kCanceled = -1001;

// Runs a process (stdout+stderr merged, lines to onLine) inside a kill-on-close
// Job Object; polls `cancel`.
int runProc(const std::vector<std::wstring>& argv, std::atomic<bool>* cancel,
            const std::function<void(const std::string&)>& onLine) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return kSpawnFailed;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = wr;
    si.hStdError = wr;
    std::wstring cmd;
    for (size_t i = 0; i < argv.size(); ++i) { if (i) cmd += L' '; cmd += quoteArg(argv[i]); }
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(L'\0');
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(argv[0].c_str(), buf.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        if (job) CloseHandle(job);
        return kSpawnFailed;
    }
    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    std::thread reader([rd, &onLine]() {
        std::string pending;
        char b[4096];
        DWORD got = 0;
        while (ReadFile(rd, b, sizeof(b), &got, nullptr) && got > 0) {
            pending.append(b, got);
            size_t nl;
            while ((nl = pending.find_first_of("\r\n")) != std::string::npos) {
                std::string line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                if (!line.empty() && onLine) onLine(line);
            }
        }
        if (!pending.empty() && onLine) onLine(pending);
    });
    bool canceled = false;
    while (WaitForSingleObject(pi.hProcess, 100) == WAIT_TIMEOUT) {
        if (cancel && cancel->load()) {
            canceled = true;
            if (job) TerminateJobObject(job, 1); else TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            break;
        }
    }
    if (job) TerminateJobObject(job, 1);    // stray grandchildren holding the pipe
    reader.join();
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(rd);
    if (job) CloseHandle(job);
    return canceled ? kCanceled : (int)code;
}

std::vector<std::wstring> curlBase(const std::string& url) {
    std::vector<std::wstring> a = {systemTool(L"curl.exe"), L"--location", L"--fail", L"--silent", L"--show-error",
                                   L"--retry", L"3", L"--retry-delay", L"3", L"--connect-timeout", L"30",
                                   // abort a stalled transfer (< 1 KB/s for 60 s); the outer loop resumes it
                                   L"--speed-limit", L"1024", L"--speed-time", L"60"};
    if (isLoopbackHttp(url)) {
        a.push_back(L"--proto"); a.push_back(L"=http");
        a.push_back(L"--noproxy"); a.push_back(L"127.0.0.1,localhost");
    }
    else {
        a.push_back(L"--proto"); a.push_back(L"=https");
        a.push_back(L"--proto-redir"); a.push_back(L"=https");
    }
    a.push_back(L"--user-agent");
    a.push_back(L"PointForge-EngineInstaller/1");
    return a;
}

// Small file -> memory (catalog / signature).
bool fetchSmall(const std::string& url, const fs::path& tmp, size_t maxBytes, std::string& out, std::string& err) {
    std::error_code ec;
    fs::create_directories(tmp.parent_path(), ec);
    fs::remove(tmp, ec);
    auto a = curlBase(url);
    a.push_back(L"--max-filesize");
    a.push_back(std::to_wstring(maxBytes));
    a.push_back(L"-o");
    a.push_back(tmp.wstring());
    a.push_back(widen(url));
    std::string last;
    const int rc = runProc(a, nullptr, [&](const std::string& l) { last = l; });
    if (rc != 0) {
        err = rc == kSpawnFailed ? "curl.exe not found (Windows 10 1803 or newer is required)"
                                 : "could not download " + url + (last.empty() ? "" : " (" + last + ")");
        fs::remove(tmp, ec);
        return false;
    }
    out = readFile(tmp, maxBytes);
    fs::remove(tmp, ec);
    if (out.empty()) { err = "empty or oversized response from " + url; return false; }
    return true;
}

// Resumable large download into `part`. onBytes gets the current size while it runs.
bool downloadResumable(const std::string& url, const fs::path& part, uint64_t expected, std::atomic<bool>& cancel,
                       const std::function<void(uint64_t)>& onBytes, std::string& err) {
    std::error_code ec;
    fs::create_directories(part.parent_path(), ec);
    std::string last;
    for (int attempt = 0; attempt < 8; ++attempt) {
        uint64_t cur = fileSize(part);
        if (cur > expected) { fs::remove(part, ec); cur = 0; }
        if (cur == expected) return true;
        auto a = curlBase(url);
        if (cur > 0) { a.push_back(L"--continue-at"); a.push_back(L"-"); }
        a.push_back(L"-o");
        a.push_back(part.wstring());
        a.push_back(widen(url));
        std::atomic<bool> done{false};
        std::thread poll([&]() {
            while (!done.load()) {
                if (onBytes) onBytes(fileSize(part));
                for (int i = 0; i < 5 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        const int rc = runProc(a, &cancel, [&](const std::string& l) { last = l; });
        done = true;
        poll.join();
        if (onBytes) onBytes(fileSize(part));
        if (rc == kCanceled) { err = "Canceled"; return false; }
        if (rc == kSpawnFailed) { err = "curl.exe not found (Windows 10 1803 or newer is required)"; return false; }
        if (rc == 0 && fileSize(part) == expected) return true;
        if (rc == 33) fs::remove(part, ec);          // server refused the byte range: start over
        if (rc == 22 && last.find("416") != std::string::npos) fs::remove(part, ec);
        logWarn("[engines] download interrupted (curl exit " + std::to_string(rc) + ": " + last + "), retrying");
        for (int i = 0; i < 30 && !cancel.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (cancel.load()) { err = "Canceled"; return false; }
    }
    err = "download failed after several attempts" + (last.empty() ? std::string() : " (" + last + ")");
    return false;
}

struct StateFile {
    fs::path path;
    explicit StateFile(const std::string& root) : path(fs::u8path(root) / ".engine-state.json") {}
    int64_t sequence() const {
        try {
            const std::string s = readFile(path, 1 << 16);
            return s.empty() ? 0 : json::parse(s).value("catalogSequence", (int64_t)0);
        } catch (...) {
            return 0;
        }
    }
    void setSequence(int64_t seq) const {
        json j = {{"catalogSequence", seq}};
        writeFileAtomic(path, j.dump(1));
    }
};

} // namespace

std::string engineRoot() {
    const wchar_t* v = _wgetenv(L"LOCALAPPDATA");
    if (!v || !*v) v = _wgetenv(L"TEMP");
    return narrow((fs::path(v ? v : L".") / L"ViitorX" / L"PointForge" / L"engines").wstring());
}

uint64_t engineFreeBytes(const std::string& root) {
    fs::path p = fs::u8path(root);
    std::error_code ec;
    while (!p.empty() && !fs::exists(p, ec)) p = p.parent_path();
    ULARGE_INTEGER freeBytes{};
    if (p.empty() || !GetDiskFreeSpaceExW(p.wstring().c_str(), &freeBytes, nullptr, nullptr)) return 0;
    return freeBytes.QuadPart;
}

bool engineSha256File(const std::string& path, std::string& hexOut, std::atomic<bool>* cancel) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return false;
    bool ok = BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    ok = ok && (bool)f;
    std::vector<char> buf(4 << 20);
    while (ok && f) {
        if (cancel && cancel->load()) { ok = false; break; }
        f.read(buf.data(), (std::streamsize)buf.size());
        const std::streamsize got = f.gcount();
        if (got > 0) ok = BCryptHashData(h, (PUCHAR)buf.data(), (ULONG)got, 0) == 0;
    }
    unsigned char digest[32];
    ok = ok && BCryptFinishHash(h, digest, sizeof(digest), 0) == 0;
    if (h) BCryptDestroyHash(h);
    BCryptCloseAlgorithmProvider(alg, 0);
    if (!ok) return false;
    char hex[65];
    for (int i = 0; i < 32; ++i) snprintf(hex + i * 2, 3, "%02x", digest[i]);
    hexOut.assign(hex, 64);
    return true;
}

bool engineVerifySignature(const std::string& data, const std::string& sigBase64, const std::string& publicKeyHex,
                           std::string& err) {
    std::vector<unsigned char> pub;
    if (publicKeyHex.size() != 128 || !hexDecode(publicKeyHex, pub)) {
        err = "this PointForge build has no engine signing key (see scripts/pf360/engine_packs.py keygen)";
        return false;
    }
    const std::string sig = base64Decode(sigBase64);
    if (sig.size() != 64) { err = "catalog signature is malformed"; return false; }

    // SHA-256 of the catalog bytes
    unsigned char digest[32];
    {
        BCRYPT_ALG_HANDLE alg = nullptr;
        if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) { err = "SHA-256 unavailable"; return false; }
        const NTSTATUS s = BCryptHash(alg, nullptr, 0, (PUCHAR)data.data(), (ULONG)data.size(), digest, 32);
        BCryptCloseAlgorithmProvider(alg, 0);
        if (s != 0) { err = "SHA-256 failed"; return false; }
    }
    BCRYPT_ALG_HANDLE alg = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_ECDSA_P256_ALGORITHM, nullptr, 0) != 0) { err = "ECDSA unavailable"; return false; }
    std::vector<unsigned char> blob(sizeof(BCRYPT_ECCKEY_BLOB) + 64);
    auto* hdr = reinterpret_cast<BCRYPT_ECCKEY_BLOB*>(blob.data());
    hdr->dwMagic = BCRYPT_ECDSA_PUBLIC_P256_MAGIC;
    hdr->cbKey = 32;
    std::copy(pub.begin(), pub.end(), blob.begin() + sizeof(BCRYPT_ECCKEY_BLOB));
    BCRYPT_KEY_HANDLE key = nullptr;
    NTSTATUS s = BCryptImportKeyPair(alg, nullptr, BCRYPT_ECCPUBLIC_BLOB, &key, blob.data(), (ULONG)blob.size(), 0);
    if (s == 0) {
        s = BCryptVerifySignature(key, nullptr, digest, 32, (PUCHAR)sig.data(), 64, 0);
        BCryptDestroyKey(key);
    }
    BCryptCloseAlgorithmProvider(alg, 0);
    if (s != 0) { err = "catalog signature is INVALID - refusing to install (catalog altered or wrong key)"; return false; }
    return true;
}

bool engineFetchCatalog(const std::string& url, const std::string& root, EngineCatalog& out, std::string& err,
                        const std::string& publicKeyHex) {
    if (url.empty()) { err = "no engine catalog URL configured (Preferences > 360 Reconstruction)"; return false; }
    if (!validUrl(url)) { err = "engine catalog URL must start with https://"; return false; }
    const std::string key = publicKeyHex.empty() ? std::string(kEnginePackPublicKeyHex) : publicKeyHex;
    if (key.size() != 128) {
        err = "this PointForge build has no engine signing key, so it cannot trust any catalog";
        return false;
    }
    const fs::path tmp = fs::u8path(root) / ".downloads" / "catalog";
    std::string body, sig;
    if (!fetchSmall(url, tmp / "catalog.json", 4u << 20, body, err)) return false;
    if (!fetchSmall(url + ".sig", tmp / "catalog.json.sig", 4096, sig, err)) return false;
    if (!engineVerifySignature(body, sig, key, err)) return false;
    if (!engineParseCatalog(body, out, err)) return false;
    const StateFile st(root);
    const int64_t seen = st.sequence();
    if (out.sequence < seen) {
        err = "the server offered an OLDER engine catalog than this PC already accepted - refusing (possible "
              "rollback attack, or the publisher re-signed an old catalog)";
        return false;
    }
    if (out.sequence > seen) st.setSequence(out.sequence);
    out.source = url;
    return true;
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------
namespace {

struct Progress {
    const EngineProgressFn& fn;
    uint64_t total = 1, base = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    uint64_t dlStart = 0;   // bytes already on disk when this session started (excluded from speed)
    void report(uint64_t doneUnits, const std::string& msg) {
        if (fn) fn(std::min(1.0f, (float)((double)doneUnits / (double)total)), msg);
    }
};

bool rollback(const fs::path& dest, const fs::path& backup, std::string& err) {
    std::error_code ec;
    fs::path failed = dest.wstring() + L".failed";
    fs::remove_all(failed, ec);
    if (fs::exists(dest, ec)) fs::rename(dest, failed, ec);
    if (!backup.empty() && fs::exists(backup, ec)) {
        fs::rename(backup, dest, ec);
        if (ec) { err += " (and the previous install could not be restored from " + narrow(backup.wstring()) + ")"; return false; }
    }
    fs::remove_all(failed, ec);
    return true;
}

bool installOne(const EnginePack& p, const fs::path& root, int64_t sequence, std::atomic<bool>& cancel, Progress& pr,
                std::string& err) {
    std::error_code ec;
    const fs::path dlDir = root / ".downloads" / fs::u8path(p.id);
    fs::create_directories(dlDir, ec);
    const std::string title = p.title.empty() ? p.id : p.title;

    // 1) download + verify every file
    for (const auto& f : p.files) {
        const fs::path final = dlDir / fs::u8path(f.name);
        const fs::path part = final.wstring() + L".part";
        if (fileSize(final) != f.size) fs::remove(final, ec);
        if (!fs::exists(final, ec)) {
            const uint64_t before = pr.base;
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t had = fileSize(part);
            auto onBytes = [&](uint64_t now) {
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double rate = secs > 1 && now > had ? (double)(now - had) / secs : 0;
                std::string msg = "Downloading " + title + "  " + fmtGB(now) + " / " + fmtGB(f.size);
                if (rate > 0) {
                    char b[64];
                    const double left = (double)(f.size - std::min<uint64_t>(now, f.size)) / rate;
                    snprintf(b, sizeof(b), "  (%.1f MB/s, ~%d min left)", rate / (1 << 20), (int)(left / 60) + 1);
                    msg += b;
                }
                pr.report(before + now, msg);
            };
            if (!downloadResumable(f.url, part, f.size, cancel, onBytes, err)) return false;
            fs::rename(part, final, ec);
            if (ec) { err = "cannot finalize download " + f.name; return false; }
        }
        pr.report(pr.base + f.size, "Verifying " + title + " (SHA-256)...");
        std::string got;
        if (!engineSha256File(narrow(final.wstring()), got, &cancel)) {
            err = cancel.load() ? "Canceled" : "cannot read " + f.name;
            return false;
        }
        if (got != f.sha256) {
            fs::remove(final, ec);
            err = f.name + " failed its SHA-256 check (corrupted download or the file changed on the server) - "
                  "it was deleted; try again";
            return false;
        }
        pr.base += f.size;
        pr.report(pr.base, "Verified " + f.name);
    }

    // 2) unpack into staging
    const fs::path staging = root / ".staging" / fs::u8path(p.id);
    fs::remove_all(staging, ec);
    fs::create_directories(staging, ec);
    const fs::path stagedDir = staging / fs::u8path(p.installDir);
    for (const auto& f : p.files) {
        const fs::path final = dlDir / fs::u8path(f.name);
        if (f.kind == "zip") {
            uint64_t n = 0;
            const uint64_t units0 = pr.base;
            const uint64_t unpackUnits = std::max<uint64_t>(p.installedBytes / 2, 1);
            pr.report(units0, "Unpacking " + title + "...");
            // bsdtar refuses absolute paths and ".." entries unless -P is given.
            const int rc = runProc({systemTool(L"tar.exe"), L"-x", L"-v", L"-f", final.wstring(), L"-C", staging.wstring()},
                                   &cancel, [&](const std::string&) {
                                       ++n;
                                       if (f.fileCount && (n % 200) == 0)
                                           pr.report(units0 + unpackUnits * std::min<uint64_t>(n, f.fileCount) / f.fileCount,
                                                     "Unpacking " + title + "  " + std::to_string(n) + " / " +
                                                         std::to_string(f.fileCount) + " files");
                                   });
            if (rc == kCanceled) { fs::remove_all(staging, ec); err = "Canceled"; return false; }
            if (rc != 0) {
                fs::remove_all(staging, ec);
                err = rc == kSpawnFailed ? "tar.exe not found (Windows 10 1803 or newer is required)"
                                         : "unpacking " + f.name + " failed (tar exit " + std::to_string(rc) + ")";
                return false;
            }
        } else {
            fs::create_directories(stagedDir, ec);
            fs::copy_file(final, stagedDir / fs::u8path(f.name), fs::copy_options::overwrite_existing, ec);
            if (ec) { fs::remove_all(staging, ec); err = "cannot stage " + f.name; return false; }
        }
    }
    pr.base += std::max<uint64_t>(p.installedBytes / 2, 1);
    if (!fs::is_directory(stagedDir, ec)) {
        fs::remove_all(staging, ec);
        err = "pack " + p.id + " did not contain " + p.installDir + "/";
        return false;
    }

    // 3) swap in (keep what was there: ours -> .old and deleted after success, anyone else's -> .backup-<time>)
    const fs::path dest = root / fs::u8path(p.installDir);
    fs::create_directories(dest.parent_path(), ec);
    fs::path backup;
    bool ownBackup = false;
    if (fs::exists(dest, ec)) {
        const EnginePackStatus st = engineStatus(narrow(root.wstring()), p);
        ownBackup = st.state == EnginePackStatus::State::Installed || st.state == EnginePackStatus::State::UpdateAvailable;
        backup = ownBackup ? fs::path(dest.wstring() + L".old")
                           : fs::path(dest.wstring() + L".backup-" + widen(nowStamp("%Y%m%d-%H%M%S")));
        fs::remove_all(fs::path(dest.wstring() + L".old"), ec);
        fs::rename(dest, backup, ec);
        if (ec) {
            fs::remove_all(staging, ec);
            err = "cannot replace " + narrow(dest.wstring()) + " - a program is using it (close running "
                  "reconstructions / diagnostics and try again)";
            return false;
        }
        if (!ownBackup) logInfo("[engines] kept the previous " + p.installDir + " as " + narrow(backup.wstring()));
    }
    fs::rename(stagedDir, dest, ec);
    if (ec) {
        err = "cannot move the unpacked " + p.id + " into place";
        rollback(dest, backup, err);
        fs::remove_all(staging, ec);
        return false;
    }
    fs::remove_all(staging, ec);

    // 4) post-install (fixed, known steps only - never a command from the catalog)
    if (p.postInstall == "python-relocate") {
        pr.report(pr.base, "Configuring " + title + "...");
        const fs::path py = dest / "python.exe";
        const fs::path script = dest / "pf360_relocate.py";
        std::string out;
        const int rc = runProc({py.wstring(), L"-I", script.wstring()}, &cancel,
                               [&](const std::string& l) { if (out.size() < 4000) out += l + "\n"; });
        if (rc != 0) {
            err = rc == kCanceled ? "Canceled" : "configuring " + p.id + " failed: " + out.substr(0, 600);
            rollback(dest, backup, err);
            return false;
        }
        logInfo("[engines] " + out);
    }

    // 5) record + clean up
    json st = {{"id", p.id}, {"version", p.version}, {"catalogSequence", sequence},
               {"installed", nowStamp("%Y-%m-%dT%H:%M:%S")}, {"files", json::array()}};
    for (const auto& f : p.files) st["files"].push_back({{"name", f.name}, {"sha256", f.sha256}});
    if (!writeFileAtomic(dest / ".pfpack.json", st.dump(1))) {
        err = "cannot write " + narrow((dest / ".pfpack.json").wstring());
        rollback(dest, backup, err);
        return false;
    }
    if (ownBackup && !backup.empty()) {
        fs::remove_all(backup, ec);
        if (ec) logWarn("[engines] could not remove the previous version at " + narrow(backup.wstring()));
    }
    fs::remove_all(dlDir, ec);
    logInfo("[engines] installed " + p.id + " " + p.version + " into " + narrow(dest.wstring()));
    return true;
}

} // namespace

bool engineInstallPacks(const EngineCatalog& cat, const std::vector<std::string>& ids, const std::string& rootUtf8,
                        std::atomic<bool>& cancel, const EngineProgressFn& progress, std::string& err) {
    if (ids.empty()) { err = "nothing selected"; return false; }
    const fs::path root = fs::u8path(rootUtf8);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) { err = "cannot create " + rootUtf8; return false; }
    const uint64_t need = engineRequiredBytes(cat, ids);
    const uint64_t freeB = engineFreeBytes(rootUtf8);
    if (freeB && freeB < need) {
        err = "not enough disk space: " + fmtGB(need) + " needed, " + fmtGB(freeB) + " free on the drive holding " +
              rootUtf8;
        return false;
    }
    Progress pr{progress};
    pr.total = 0;
    std::vector<const EnginePack*> packs;
    for (const auto& id : ids) {
        const EnginePack* p = cat.find(id);
        if (!p) { err = "pack not in catalog: " + id; return false; }
        packs.push_back(p);
        pr.total += p->downloadBytes() + std::max<uint64_t>(p->installedBytes / 2, 1);
    }
    pr.total = std::max<uint64_t>(pr.total, 1);
    for (const EnginePack* p : packs) {
        if (cancel.load()) { err = "Canceled"; return false; }
        if (!installOne(*p, root, cat.sequence, cancel, pr, err)) {
            if (err != "Canceled") err = (p->title.empty() ? p->id : p->title) + ": " + err;
            return false;
        }
    }
    fs::remove_all(root / ".staging", ec);
    if (progress) progress(1.0f, "Engines installed");
    return true;
}

#else   // ------------------------------------------------------------------ non-Windows stubs
std::string engineRoot() { return {}; }
uint64_t engineFreeBytes(const std::string&) { return 0; }
bool engineSha256File(const std::string&, std::string&, std::atomic<bool>*) { return false; }
bool engineVerifySignature(const std::string&, const std::string&, const std::string&, std::string& err) {
    err = "engine packs are Windows-only";
    return false;
}
bool engineFetchCatalog(const std::string&, const std::string&, EngineCatalog&, std::string& err, const std::string&) {
    err = "engine packs are Windows-only";
    return false;
}
bool engineInstallPacks(const EngineCatalog&, const std::vector<std::string>&, const std::string&, std::atomic<bool>&,
                        const EngineProgressFn&, std::string& err) {
    err = "engine packs are Windows-only";
    return false;
}
#endif

} // namespace pf
