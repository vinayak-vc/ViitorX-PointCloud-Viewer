#pragma once
// Downloadable engine packs ("DLC") for the 360 Reconstruction Studio.
//
// The viewer ships without the ~11 GB reconstruction engines. They are listed
// in a SIGNED catalog (catalog.json + catalog.json.sig, ECDSA P-256/SHA-256,
// public key compiled in from EnginePackKey.h) hosted on Hugging Face, and
// installed on demand into %LOCALAPPDATA%\ViitorX\PointForge\engines:
//
//   pf360-runtime -> engines\pf360-env      (Python + PyTorch + Nerfstudio)
//   colmap        -> engines\colmap-conda
//   ffmpeg        -> engines\ffmpeg
//   mask-model    -> engines\models\upernet-convnext-base-ade20k (optional)
//
// Trust chain: the catalog must carry a valid signature from the embedded key
// and a sequence number not older than the last accepted one; every file it
// lists is pinned by size + SHA-256 and checked before anything is unpacked or
// run. Downloads use the system curl.exe (HTTPS only, resumable), archives are
// unpacked with the system tar.exe into a staging folder, then swapped in.
// Built by scripts/pf360/engine_packs.py; format in docs/360-reconstruction.md.
//
// Viewer-only, Windows-only (other platforms get stubs that report "unsupported").
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pf {

struct EnginePackFile {
    std::string name, kind, url, sha256;   // kind: "zip" (unpacked) or "file" (copied as is)
    uint64_t size = 0;
    uint64_t fileCount = 0;                // zip entries (progress only; 0 = unknown)
};

struct EnginePack {
    std::string id, title, description, version, group, installDir, postInstall, license;
    bool required = false;
    uint64_t installedBytes = 0;
    std::vector<EnginePackFile> files;
    uint64_t downloadBytes() const {
        uint64_t n = 0;
        for (const auto& f : files) n += f.size;
        return n;
    }
};

struct EngineCatalog {
    int64_t sequence = 0;
    std::string created, source;           // source = URL it was fetched from
    std::vector<EnginePack> packs;
    const EnginePack* find(const std::string& id) const {
        for (const auto& p : packs) if (p.id == id) return &p;
        return nullptr;
    }
};

// What is on disk for one pack.
struct EnginePackStatus {
    enum class State { Missing, Installed, UpdateAvailable, SelfManaged };
    State state = State::Missing;
    std::string installedVersion;          // from the pack's .pfpack.json
};

// %LOCALAPPDATA%\ViitorX\PointForge\engines (UTF-8).
std::string engineRoot();
// Build-time default catalog URL (CMake PF_ENGINE_CATALOG_URL); "" = not configured.
std::string engineDefaultCatalogUrl();
// True when this build embeds a catalog-signing public key.
bool engineHasSigningKey();

// Downloads catalog.json + catalog.json.sig from `url` (https, or http://127.0.0.1 /
// localhost for tests), verifies the signature with `publicKeyHex` ("" = the
// compiled-in key) and parses it. Rejects catalogs older than the newest one
// accepted under `root` (rollback protection) and records the new sequence.
// Blocking network call: background thread only.
bool engineFetchCatalog(const std::string& url, const std::string& root, EngineCatalog& out, std::string& err,
                        const std::string& publicKeyHex = "");

// Pieces of the above, exposed for tests.
bool engineVerifySignature(const std::string& data, const std::string& sigBase64, const std::string& publicKeyHex,
                           std::string& err);
bool engineParseCatalog(const std::string& json, EngineCatalog& out, std::string& err);
bool engineSha256File(const std::string& path, std::string& hexOut, std::atomic<bool>* cancel = nullptr);

EnginePackStatus engineStatus(const std::string& root, const EnginePack& pack);
// Free bytes on the volume holding `root` (0 = unknown).
uint64_t engineFreeBytes(const std::string& root);
// Bytes needed on disk to install `ids` (downloads + unpacked size, with margin).
uint64_t engineRequiredBytes(const EngineCatalog& cat, const std::vector<std::string>& ids);

using EngineProgressFn = std::function<void(float fraction, const std::string& message)>;

// Downloads (resuming partial files), verifies, unpacks and swaps in each pack in
// `ids`. An existing folder this installer did not create is kept as
// "<dir>.backup-<time>", never deleted. On failure the previous install is
// restored. Blocking: run on the job thread.
bool engineInstallPacks(const EngineCatalog& cat, const std::vector<std::string>& ids, const std::string& root,
                        std::atomic<bool>& cancel, const EngineProgressFn& progress, std::string& err);

} // namespace pf
