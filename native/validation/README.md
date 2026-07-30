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

The production depth path always uses the empty prompt. A later coherent
derived representation may replace the 1.36 GB CLIP component with its
validated constant empty-prompt embedding, keyed by all three canonical
hashes and converter version. The canonical snapshot remains shared with
the Python backend.

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

The single-step conditional UNet and end-to-end numerical comparison remain
to be implemented before this backend can advertise inference capability.
