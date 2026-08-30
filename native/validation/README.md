# Lotus native validation

## Canonical model boundary

InferBridge's default Lotus entry is the generation checkpoint
`jingheya/lotus-depth-g-v1-0` at revision
`d8f1b0835745cb1ef5c8b89528c8d37bc764a37b`.

The dependency-free native loader consumes the snapshot's canonical
Safetensors files directly. It does not parse pickle or import Python:

| Component | Tensors | Bytes | SHA-256 |
|---|---:|---:|---|
| UNet | 690 | 3,470,357,352 | `41a8d50a989a757016251e170f39b6ac90e1ff324a90ae88c9a762ec549d0f34` |
| VAE | 248 | 334,643,268 | `fc436097e5cbb105c44df4403ae84acf7e81de1f64acfedc334888e9a8eea223` |
| CLIP text encoder | 372 | 1,361,596,304 | `8a0859e9385019944df829693ffdef5bbde394950ff561cbfdd61fbb3ea76fb5` |

The memory-mapped reader validates the Safetensors header, tensor names,
dtypes, dimensions, byte counts, offsets, non-overlap, and full payload
coverage before exposing any tensor.

The production depth path always uses the empty prompt. The native converter
therefore derives a 315,904-byte `LOTUSP01` cache containing only the validated
77x1024 empty-prompt embedding. Its header binds the payload to the snapshot
revision and all three canonical hashes. The DLL maps only the canonical UNet
and VAE at runtime, avoiding the 1.36 GB CLIP mapping without creating a
second model entry or weight download. The canonical snapshot remains shared
with the Python backend.

## Current gate

The model probe validates all three canonical files and architecture anchor
shapes. `dump_reference.py` also produces a deterministic Python CPU fixture
at 64x64 with explicit posterior and initial noise. Its component anchors are:

| Tensor | Minimum | Maximum |
|---|---:|---:|
| VAE posterior mean | `-16.4733543` | `15.0739040` |
| Scaled RGB latent | `-3.00059485` | `2.74570560` |
| One-step UNet prediction | `-1.82667851` | `1.16818595` |
| Decoded depth | `0.0287314560` | `0.547842979` |

The dependency-free CPU VAE now passes the Python CPU fixture:

| Component | Relative L1 | Maximum absolute |
|---|---:|---:|
| Posterior mean | `8.27883e-7` | `2.14577e-5` |
| Posterior log variance | `1.10318e-6` | `5.91278e-5` |
| Decoder RGB | `5.85769e-7` | `3.45707e-6` |

The single-step conditional UNet passes with relative L1 `7.24859e-6` and
maximum absolute error `2.18749e-5`.

## Full native DLL gate

`lotus_native.dll` exposes ABI 3 lifecycle and inference calls. The exact
validation entry accepts explicit initial and VAE-posterior noise, removing
stochastic ambiguity while comparing the complete graph against Python CPU.
The normal entry owns a stable seeded native RNG.

ABI 3 adds the InferBridge image contract without changing the tensor ABI:
`lotus_inferbridge_image_shape` reports the model's 768-pixel longest-edge
working shape, while `lotus_infer_bgra8_f32[_with_noise]` retains the BGRA
source's first three channels in BGR order, applies the Python harness's
nearest-neighbour preprocessing and matching resize, and performs final
per-image min/max normalization. The byte-image path is host-backed and does
not claim external-resource or zero-copy GPU support.

| Gate | Result |
|---|---:|
| Full 64x64 relative depth L1 | `6.81436e-6` (`0.000681%`) |
| Full maximum absolute depth error | `1.69128e-5` |
| Non-multiple input | `65x73` passed |
| C ABI smoke test | passed |

The additive `lotus_create_vulkan` entry maps the same canonical UNet and VAE
files into a dependency-free full Vulkan graph and fails rather than falling
back to CPU. RGB upload and final depth download remain at the tensor ABI
boundary; VAE encoding, posterior sampling, conditional UNet, VAE decoding,
and every intermediate stay on the selected GPU.

| GPU | Full 64x64 relative L1 | Maximum absolute |
|---|---:|---:|
| Radeon RX 9070 | `7.33747e-6` | `1.19284e-5` |
| GeForce GTX 1080 | `8.36691e-6` | `2.21133e-5` |
| Radeon RX 6700 XT | `6.97460e-6` | `1.29789e-5` |

Five consecutive calls on persistent contexts also passed. Concurrent canary
medians were 569.98 ms (RX 9070), 711.28 ms (GTX 1080), and 327.01 ms
(RX 6700 XT); they are stability measurements for this untuned FP32 graph,
not isolated comparative benchmarks.

This is the FP32 correctness baseline. Mixed precision and external
GPU-resource import/export are not advertised until separate accuracy and
interop gates pass.

## First performance pass

VAE and UNet ResNet, attention, and transformer composites now use bounded
Vulkan command batches. Buffer snapshots made inside a batch are recorded with
explicit transfer/compute barriers. Batches remain block-sized so the runtime
does not replace the correctness-first Windows watchdog bounds with one
monolithic submission.

The isolated RX 9070 64x64 median improved from `454.6 ms` to `191.9 ms`
(`57.8%`) in matched seven-iteration runs. Five-call validation after the
change passed on all three adapters with unchanged numerical results; observed
medians were `246.5 ms` (RX 9070), `630.4 ms` (GTX 1080), and `214.3 ms`
(RX 6700 XT).

## Wave32 packed-weight pass

The Wave32 path packs linear and 3x3 convolution weights in input-major,
output-contiguous FP16 storage while retaining FP32 activation and
accumulation. Normalization plus SiLU is fused, ResNet residuals no longer
make redundant device copies, and the shared timestep activation is computed
once per inference. The pointwise and linear tiles also use bank-safe shared
memory layouts.

For the discriminative 64x64 fixture, isolated persistent-context medians
were `96.35 ms` on Radeon RX 9070, `106.621 ms` on GeForce GTX 1080
(101 calls), and `95.981 ms` on Radeon RX 6700 XT. The final relative L1
deviation was `0.000629` (`0.0629%`) on all three adapters, with maximum
absolute deviation below `0.00157`.

## Discriminative checkpoint

The same ABI now detects and executes InferBridge's selectable regression
checkpoint `jingheya/lotus-depth-d-v1-0` at revision
`9b858ffdbec93117d50de899e8bccf64345d8f3b`. Its canonical UNet SHA-256 is
`66f32e128f0f85a6f2d72893f56c663f8abde5a76678180b0eb51b1f3ed44899`;
the VAE and text encoder are byte-identical to the generative checkpoint.
The regression path samples the VAE posterior, feeds its four-channel RGB
latent directly to the discriminative UNet, and decodes that prediction. It
does not manufacture or consume a generative target latent.

The regression prompt sidecar reuses the identical text-encoder result but
has distinct content-bound revision and UNet metadata. The loader rejects a
generative sidecar for a regression model and vice versa.

| Executor | Full 64x64 relative L1 | Maximum absolute |
|---|---:|---:|
| CPU | `3.72883e-6` | `2.51532e-5` |
| Radeon RX 9070 | `4.17826e-6` | `1.32918e-5` |
| GeForce GTX 1080 | `4.25895e-6` | `9.23872e-6` |
| Radeon RX 6700 XT | `4.11891e-6` | `1.37687e-5` |

The 768x64 BGRA image-contract equivalence canary also passed on all three
GPUs with exactly zero maximum and mean absolute difference versus the
validated tensor path plus final normalization.

## Embedded InferBridge harness

The model DLL exports `ibrh_get_api` for InferBridge harness ABI 2.0. In
addition to the host-memory compatibility path, it consumes Core-owned shared
D3D12 BGRA8/RGBA8 input textures and wait fences and writes source-sized
normalized depth directly to a Core-owned shared D3D12 R32_FLOAT texture. Core
owns transfer resources and output leases; the harness borrows the handles for
the job lifetime.

`model_path` is the canonical generation or regression snapshot selected for
the single Lotus model entry. `PromptCache` in the model parameters names the
small, hidden, content-bound `.lotusp` derivation for that snapshot. This
keeps the Python and Native backends on the same canonical Safetensors files;
the prompt sidecar is not a second model card or weight download. A
generation sidecar cannot be used with regression weights or vice versa.

The Python template passes no generator and is therefore intentionally
stochastic. The harness accepts an optional unsigned `Seed` parameter for
reproducible runs. Otherwise it uses `source_frame_id`, then timestamp, then
a per-model sequence. Exact numerical validation continues to use the
explicit-noise ABI, which removes RNG-algorithm ambiguity and validates the
actual graph.

The external path imports the exact requested adapter LUID, preprocesses the
texture and generates seeded noise on Vulkan, executes the graph in bounded
device-only stages, and signals only the supplied final fence. Its persistent
worker makes public submission asynchronous and admits at most three jobs.
Prompt, timestep, and class constants are uploaded once during model load;
per-frame transfer counters remain unchanged. It uses no queue-wide idle call
and never falls back to a host pixel/depth path under the GPU capability.

The Windows Release ABI2 full-graph gate passes both snapshots on the RX 9070
with Core-owned input/output resources, explicit fences, source-frame
correlation, finite varying R32 output, and zero per-frame upload/readback:

| Variant | Relative L1 | Maximum absolute | Submit return | Upload/download delta |
|---|---:|---:|---:|---:|
| Generation | `0.000950690` | `0.00248301` | `0.0082 ms` | `0 / 0` bytes |
| Regression | `0.000628796` | `0.00156927` | `0.0064 ms` | `0 / 0` bytes |

The external 64x64 request deliberately exercises the production 768-pixel
longest-edge processing shape; each isolated ABI2 job completed in about 21
seconds on the RX 9070. The bounded staging is a correctness and residency
gate, not a performance claim.

## Native Metal executor

The macOS harness now ports the same canonical VAE encoder, posterior sample,
single-step conditional UNet, and VAE decoder directly to MPSGraph. Both the
eight-channel generative checkpoint and four-channel discriminative checkpoint
use the same public ABI and keep learned intermediates on Metal. The port
preserves the reference's asymmetric VAE downsampling, skip-aligned nearest
upsampling, group/layer normalization epsilons, single-head VAE attention, and
multi-head UNet self/cross attention.

| Variant / precision | Full 64x64 relative L1 | Maximum absolute |
|---|---:|---:|
| Generation FP32 | `5.16004e-6` | `1.62721e-5` |
| Regression FP32 | `2.46795e-6` | `1.10269e-5` |
| Generation FP16 | `0.00338574` | `0.0075831` |
| Regression FP16 | `0.00449251` | `0.0131227` |

Persistent, model-identity-bound MPSGraph packages remove graph compilation
from later starts. On the Apple M1 validation machine, warm generation FP32
and FP16 medians at 64x64 were about 177 ms and 104 ms respectively. A 65x73
non-multiple tensor input and the real InferBridge host-transfer lifecycle at
the production diffusion preprocessing shape also passed for both variants.
