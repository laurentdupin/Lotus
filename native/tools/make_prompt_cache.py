"""Create the small, content-bound native empty-prompt cache."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

VAE_SHA = "fc436097e5cbb105c44df4403ae84acf7e81de1f64acfedc334888e9a8eea223"
TEXT_SHA = "8a0859e9385019944df829693ffdef5bbde394950ff561cbfdd61fbb3ea76fb5"
MODELS = {
    "generation": (
        "d8f1b0835745cb1ef5c8b89528c8d37bc764a37b",
        "41a8d50a989a757016251e170f39b6ac90e1ff324a90ae88c9a762ec549d0f34",
    ),
    "regression": (
        "9b858ffdbec93117d50de899e8bccf64345d8f3b",
        "66f32e128f0f85a6f2d72893f56c663f8abde5a76678180b0eb51b1f3ed44899",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def put(header: bytearray, offset: int, size: int, text: str) -> None:
    encoded = text.encode("ascii") + b"\0"
    if len(encoded) > size:
        raise ValueError("metadata field is too long")
    header[offset : offset + len(encoded)] = encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--variant", choices=MODELS, required=True)
    parser.add_argument(
        "--reuse-prompt-cache",
        type=Path,
        help=(
            "reuse an existing LOTUSP01 payload; safe only because both "
            "pinned Lotus variants have the identical text encoder"
        ),
    )
    args = parser.parse_args()
    root = Path(args.snapshot)
    revision, unet_sha = MODELS[args.variant]
    files = {
        "UNet": (
            root / "unet" / "diffusion_pytorch_model.safetensors",
            unet_sha,
        ),
        "VAE": (
            root / "vae" / "diffusion_pytorch_model.safetensors",
            VAE_SHA,
        ),
        "text encoder": (
            root / "text_encoder" / "model.safetensors",
            TEXT_SHA,
        ),
    }
    for label, (path, expected) in files.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )

    if args.reuse_prompt_cache:
        source = args.reuse_prompt_cache.read_bytes()
        if (
            len(source) != 512 + 77 * 1024 * 4
            or source[:8] != b"LOTUSP01"
            or struct.unpack_from("<IIII", source, 8)
            != (1, 512, 77, 1024)
        ):
            raise RuntimeError("invalid source Lotus prompt cache")
        prompt_payload = source[512:]
    else:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            root, subfolder="tokenizer", local_files_only=True
        )
        encoder = CLIPTextModel.from_pretrained(
            root, subfolder="text_encoder", local_files_only=True
        ).eval()
        tokens = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            prompt = encoder(
                tokens, return_dict=False
            )[0].contiguous().float()
        prompt_payload = prompt.numpy().tobytes()

    header = bytearray(512)
    header[:8] = b"LOTUSP01"
    struct.pack_into("<IIII", header, 8, 1, 512, 77, 1024)
    put(header, 24, 41, revision)
    put(header, 65, 65, unet_sha)
    put(header, 130, 65, VAE_SHA)
    put(header, 195, 65, TEXT_SHA)
    Path(args.output).write_bytes(header + prompt_payload)


if __name__ == "__main__":
    main()
