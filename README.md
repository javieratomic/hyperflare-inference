<p align="center">
  <img src="docs/images/hyperflare.png" alt="Flare — ASCII wordmark above a supermassive solar flare" width="912">
</p>

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

## About this fork and branch

This is a community fork of [Hyperflare](https://github.com/NVIDIA/hyperflare-inference),
a local inference server for NVIDIA SM120/SM121 GPUs published by NVIDIA as a
research prototype. It is not an official NVIDIA release. The text below this
section is upstream's README, unchanged.

This `integration` branch is upstream `main` at `f9aec99c` with the seven
changes below merged on top. Each one lives in its own branch of this fork, is
merged here as it stands there, and is described upstream in a pull request, an
issue or an issue comment. None is merged upstream yet.

### What this branch adds

Measured with DeepSeek V4.1 Flash at tensor parallelism 3 on three DGX Spark
class (GB10) hosts, one rank per host; most figures are one run per
configuration. The tables below link each figure to the item that reports it,
with its baseline, conditions and limits.

- **More KV cache.** `--system-headroom SIZE` (or `[memory] system_headroom`)
  sets how much memory cache sizing leaves to the system on unified-memory
  hosts. With `4G` in place of the default (a twentieth of memory, 6.08 GiB on
  these hosts), the KV cache holds 10.68 M tokens instead of 8.17 M (+31 %);
  `2G` gives 13.06 M and `1G` 14.36 M, at the cost noted under Serving. The
  default is unchanged, so the gain needs the option, given to every rank in
  these runs.
- **Faster cold prefill.** Engram rows are read from disk ahead of prefill on
  every rank. A cold prompt of 120,000 random token IDs takes 32.1 s instead of
  39.5–40.4 s, and four at once 134.7 s instead of 168.2–171.6 s. Random IDs
  are the case with the fewest row-cache hits; a 100,000-token natural-text
  prompt took 26.7 s against 29.9 s. There is no option: it is on for Linux
  disk tables that have a row cache and an XOR-modulo hash.
- **Faster multi-host preparation.** Five changes shorten phases of a multi-host
  export or let it proceed with less free memory: peers read their own
  checkpoint copy, received payloads are linked into the core cache, Engram
  planes are hashed in parallel, core payloads are published from a
  largest-first queue, and the compiler memory headroom is a setting.

The two serving changes were also measured together, on a build of `6b68b9b4`
with both and `--system-headroom 4G` (local runs, not reported upstream): two
starts held 10.5 and 10.7 M KV tokens and took 32.1 and 32.5 s for a cold prompt
of 120,000 random token IDs and 138.5 and 136.3 s for four at once.

### Serving

| Change | Branch | Upstream | Measured | Before | After |
|---|---|---|---|---|---|
| Disk-backed Engram rows are read ahead of prefill on every rank; no option, on for Linux disk tables with a row cache and an XOR-modulo hash | `engram-prefetch` | [#334](https://github.com/NVIDIA/hyperflare-inference/pull/334) (draft), measured in a [comment on #267](https://github.com/NVIDIA/hyperflare-inference/issues/267#issuecomment-6953943209) | Cold 120,000-token prompt of random token IDs: one prompt, and four at once | 39.5–40.4 s, and 168.2–171.6 s | 32.1 s, and 134.7 s |
| `--system-headroom SIZE` / `[memory] system_headroom`: memory that cache sizing leaves to the system on unified-memory hosts; the default stays one twentieth of memory (6.08 GiB on these hosts) | `system-headroom` | [#323](https://github.com/NVIDIA/hyperflare-inference/pull/323) | KV-cache tokens | 8.17 M | 10.68 M at `4G`, 13.06 M at `2G`, 14.36 M at `1G` |

A smaller system headroom makes the server start more slowly and keep more of
its memory in swap; #323 reports both. The read-ahead evaluates the Engram hash
natively in Helios. Whether Helios may do that is the decision #334 asks the
maintainers for, which is why it is a draft; the comment on #267 compares the
change with the `topic/engramcache` branch that the issue proposes for merging.

### Preparation

| Change | Branch | Upstream | Phase | Before | After |
|---|---|---|---|---|---|
| `ignition join --local-checkpoint [require\|prefer]`: peers read their own copy of the checkpoint | `local-checkpoint` | [#126](https://github.com/NVIDIA/hyperflare-inference/issues/126) (issue) | Load weights on the peers | 252 s | 44.7–50.1 s |
| Link received payload objects into the core cache instead of copying them | `receive-adopt` | [#125](https://github.com/NVIDIA/hyperflare-inference/pull/125) | Receive rank export payloads | 457.8 s | 280.6 s |
| Hash Engram planes on parallel readers | `parallel-engram-digest` | [#157](https://github.com/NVIDIA/hyperflare-inference/pull/157) | Assemble core payloads | 262.0 s | 169.1 s |
| Publish core cache payloads on a largest-first work queue and read ahead in slice digests | `parallel-publication` | [#124](https://github.com/NVIDIA/hyperflare-inference/pull/124) | Export native cores, with an empty core cache | 2132 s | 1692 s |
| The same | `parallel-publication` | [#124](https://github.com/NVIDIA/hyperflare-inference/pull/124) | Export native cores, with the previous core cached | 1340 s | 822 s |
| `--compile-headroom SIZE` / `[preparation] compile_headroom`: system memory that compiler processes and kernel tuning leave free; the default stays 8 GiB | `compile-headroom` | [#122](https://github.com/NVIDIA/hyperflare-inference/pull/122) | — | — | No time is claimed. With `4G`, an export passed the point where the 8 GiB floor leaves kernel tuning a 0-byte scratch limit ([#270](https://github.com/NVIDIA/hyperflare-inference/issues/270)). A lower value is also expected to admit a forward compiler on hosts where the 8 GiB floor admits none ([#242](https://github.com/NVIDIA/hyperflare-inference/issues/242)); unit tests cover that case and no export reproduced it |

The #125 figures were measured on an earlier tree with an earlier version of
that change; the #157 figures on builds based on `a161b17f` that also contained
earlier versions of #122, #124, #125 and the #126 branch; and the #124 figures
on upstream `main` at `d2b5c5f3`. None of the three was measured again after
being rebased. With `--compile-headroom 4G` the export of #270 completed, with
the memory pressure and swap use that #122 reports.

### What was measured and checked where

The before and after figures of each change were measured on an earlier
upstream revision, in some cases together with earlier versions of other
changes listed here; none was measured on this branch.

What was checked on this branch: it builds (`flare-cli` in release mode with
`--features roce`), and its unit suites fail only in tests that also fail on
upstream `main` at `f9aec99c` in the same environment, with one exception: four
`flare` hub tests failed with a refused connection to their local mock server in
a run made while other suites were running, and passed when that suite ran
alone. The branch does not change `flare/src/hub.rs`, which holds them. The
suites are the Rust tests of selene, selene-manager, helios (without a GPU),
heliopause, flare and flare-cli, and the Python tests under `tests/cli` and
`tests/preparation` with the recipe tests #122 lists. A build of this branch
then served, for one start with `--system-headroom 4G` and the throughput
scheduler policy, a core of that setup prepared from `6b68b9b4` plus #122: a
cold 120,000-token prompt took 32.8 s and four at once 141.5 s, with no
allocation error and 9.93 M KV tokens. Two starts of the `6b68b9b4` build with
both serving changes and the same options held 10.5 and 10.7 M; before this
start the host with the least free memory had 0.4 to 0.5 GiB less available than
before those two. No export was run on this branch, so the preparation changes
were exercised here only by their unit tests.

Hyperflare is a local inference server for NVIDIA SM120/SM121 GPUs, built for
long conversations, coding agents, and repeated tool use. Its goal is to make
the memory, bandwidth, and compute in workstations and compact systems such
as DGX Spark go further, while keeping generation responsive.

This is a research prototype, and is made available via open-source to enable
technical collaboration and collect early feedback.

The design rests on a few principles:

- **Embrace the standard PyTorch ecosystem at build time; serve from Rust.**
  Ignition uses PyTorch, Transformers, and FlashInfer/b12x to load models,
  tune kernels, compile forwards, and capture CUDA graphs into a runtime core.
  Helios, the Rust runtime, serves that core without Python. Compilation and
  graph capture finish before serving.

- **Build on the model ecosystem.** Reuse canonical model implementations
  wherever their dataflow fits, and specialize the parts that need different
  layouts or execution. Model development keeps the tools and libraries that
  make the Python ecosystem productive.

- **Make memory and state explicit.** Plan buffers and execution capacities
  ahead of time. Treat conversation state—including recurrent state alongside
  KV caches—as a whole, so reuse, branching, and resuming preserve it correctly.

- **Keep the runtime small and shared.** Model recipes own model computation;
  the runtime owns scheduling, cache storage, and request handling. A focused
  hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Hyperflare trades that breadth for a focused
serving contract around selected models and local workloads. Hyperflare is not, and
will never be, a replacement for production inference engines like vLLM.

Think of Hyperflare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md).
For native Windows development and serving prebuilt model cores, follow
[Windows serving](docs/windows.md#build-and-test-on-windows).
See the [CLI guide](docs/cli-design.md) for configuration and the
[core design](docs/core-design.md) for the architectural reasoning.

## Licensing and contributions

Hyperflare's project code is licensed under [Apache 2.0](LICENSE). Contributions
require the sign-off described in [CONTRIBUTING.md](CONTRIBUTING.md).
Third-party attributions are listed in [NOTICE](NOTICE); upstream terms remain
applicable to third-party code and assets.

The PyTorch-derived image resizing implementation in `helios/` retains the
[PyTorch BSD-style license](helios/licenses/PyTorch.txt), also distributed with
[Heliopause](heliopause/licenses/PyTorch.txt). The DeepSeek recipe assets retain
their [MIT license](ignition/recipes/deepseek41/assets/LICENSE). The
`heliopause/LICENSE` and `heliopause/NOTICE` symlinks refer to the root files for
Python package distribution.

Upstream chat templates are fetched with the model checkpoints and annotated
locally using bundled insertion patches. Gemma uses pinned Google sources.
The custom Kimi segmented renderer is included as project code. Downloaded
model assets and exported model cores retain their upstream terms; see
[frontend preparation](docs/model-development.md#sampling-and-frontend-metadata).
