# llama.cpp Binary Review — Current vs Latest

*Generated 2026-06-16*

## 1. Current binary (this repo)

| Field | Value |
|---|---|
| Build | **b9209** |
| Commit | `0caf2a1d` (`0caf2a1d48d2b678f2ea2fdfcb07ee35816f9f5e`) |
| Commit date | **2026-05-18** |
| Compiler | Clang 19.1.5 |
| Target | Windows x86_64 |
| Package type | Official **Windows CUDA** pre-built release |

The directory `llama.cpp_latest/` contains the standard GitHub pre-built layout:

- `ggml-cuda.dll` — CUDA GPU backend (the workhorse for this deployment)
- `ggml-rpc.dll` — RPC backend
- **15 per-microarchitecture CPU variants** chosen at load time via runtime dispatch:
  `ggml-cpu-x64`, `-sse42`, `-sandybridge`, `-ivybridge`, `-haswell`, `-skylakex`,
  `-icelake`, `-cascadelake`, `-cooperlake`, `-sapphirerapids`, `-alderlake`,
  `-cannonlake`, `-zen4`, `-piledriver`, `-base`

This is build **b9209**, the binary CLAUDE.md pins as the first to support MTP / built-in speculative decoding (PR ggml-org/llama.cpp#22673).

## 2. Latest release on GitHub (ggml-org/llama.cpp)

| Field | Value |
|---|---|
| Build | **b9670** |
| Date | **2026-06-16** |
| Headline change | "Fix and restrict NVFP4 edge-cases in llama-graph" |

The project publishes a tagged release on essentially every merged commit, so the
"latest" tag advances dozens of times per day. The meaningful axis is the commit
range, not the individual tag.

## 3. Gap: current vs latest

- **462 commits behind** master (`0caf2a1d...master`, ahead_by 462, behind_by 0).
- **~1 month** of development (2026-05-18 → 2026-06-16).

### Notable changes in that window (categorized from commit log)

- **Speculative decoding / MTP** (directly relevant — this deployment uses MTP):
  - Extracted speculative max-draft-size logic out of the server handler.
  - Removed the `draft-simple` auto-enable path; common speculative fixes.
  - `arg: fix double mtp download` and `models: fix Step3.5 MTP` — argument/parsing
    fixes around `--spec-type draft-mtp` and draft-N handling.
- **Server / WebUI**:
  - Real-time reasoning streaming in the server + WebUI ("server: real-time reasoning").
  - SSE pipeline additions; server origin refactor; tool-selector toggle fixes;
    accessibility (keyboard nav) fixes; custom CSS injection; chat-template "think" support.
  - HTTP header additions in `server-http.h`.
- **Performance** (CUDA / Vulkan / Metal matmul kernels):
  - Multiple matmul/quant kernel improvements reporting ~24%, ~57%, ~78% speedups
    in specific code paths (block-load switch, post-GEMM handling).
- **New model architectures**: EXAONE 4.5, Mellum, Step3.5/Step3.7, plus GLU/Metal
  kernel templating and multimodal (`build_vit()` skip) work.
- **NVFP4** dequantization / LoRA / bias correctness fixes (the b9670 headline).

> No security CVE or data-loss fix was identified in the range — the gap is
> feature/perf/correctness, not an urgent must-patch.

### Should you upgrade?

- **For this deployment specifically**: the MTP argument-parsing fixes and the
  real-time-reasoning server work are the most relevant items, plus CUDA matmul
  perf. None are urgent, but an upgrade is low-risk and would pick up ~1 month of
  GPU-kernel speedups and MTP robustness fixes. Validate the custom
  `chat_template.jinja` + `preserve_thinking` kwarg still behave after upgrading,
  since server template handling changed in this window.

## 4. Build from source vs pre-compiled (Windows)

### Pre-compiled GitHub release (what this repo uses)

- **Zero build toolchain needed.** Download the `-bin-win-cuda-x64.zip`, unzip, run.
  (CUDA builds additionally need the matching `cudart` redistributable on PATH.)
- **CPU-feature optimization is already solved.** The release ships the 15
  `ggml-cpu-*.dll` microarch variants above and selects the best match for the host
  CPU at runtime. So AVX2 / AVX-512 / AMX paths are *already used* — building from
  source with `-DGGML_NATIVE=ON` yields essentially **no CPU speedup** over the
  pre-built package.
- **Built with Clang** (19.1.5), which historically produces ggml CPU kernels at
  least as fast as MSVC.
- Tracks tagged builds; you cannot pick an arbitrary mid-day commit without waiting
  for its tag (though tags are near-continuous).

### Building from source

- **Toolchain required**: CMake + a compiler (MSVC from Visual Studio Build Tools,
  or Clang), **CUDA Toolkit** (nvcc) for the GPU backend, and/or the Vulkan SDK for
  the Vulkan backend. CUDA compiles are slow (many nvcc TUs across SM arches).
- **When it actually helps**:
  - You need a **commit newer than the latest tag**, or a PR not yet merged.
  - You want a **backend not in the standard release** (e.g. a custom CUDA arch list
    via `CMAKE_CUDA_ARCHITECTURES` to trim build to your exact GPU, ROCm/HIP, SYCL).
  - You want to enable/disable specific features (e.g. `-DGGML_CUDA_F16`,
    custom flash-attention flags) or patch the source.
- **Performance reality for this deployment**: inference here runs on the **GPU via
  CUDA**, so the GPU kernels do the heavy lifting and host-CPU SIMD flags are nearly
  irrelevant. A from-source build targeting only your GPU's compute capability can
  cut binary size and build time, but **runtime token throughput is dominated by the
  CUDA kernels, which are identical to the pre-built ones** at the same commit.

### Backends comparison

| Backend | In pre-built CUDA zip | Notes |
|---|---|---|
| CPU (multi-arch) | ✅ runtime-dispatched | Already optimal; no source build needed |
| CUDA | ✅ `ggml-cuda.dll` | Needs `cudart` redist; this deployment's path |
| Vulkan | ❌ (separate zip) | Build from source or grab the `-vulkan-` release |
| ROCm/HIP, SYCL | ❌ | Source build only on Windows |
| RPC | ✅ `ggml-rpc.dll` | Distributed inference |

### Recommendation

**Stay on pre-built releases.** For a CUDA + Windows deployment that already gets
runtime CPU-feature dispatch, building from source offers no meaningful speedup and
adds toolchain maintenance burden. Reserve source builds for (a) needing an
un-tagged/PR commit, or (b) a backend the release doesn't ship. To close the current
1-month gap, simply download the latest `b96xx` CUDA Windows zip and swap the
`llama.cpp_latest/` contents (after validating the custom chat template still works).

---

*Sources: local `llama-server.exe --version`; GitHub ggml-org/llama.cpp releases
page and Compare API (`0caf2a1d...master`); binary directory backend DLL listing.*
