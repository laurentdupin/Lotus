"""Create the small, content-bound native empty-prompt cache."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer


REVISION = "d8f1b0835745cb1ef5c8b89528c8d37bc764a37b"
UNET_SHA = "41a8d50a989a757016251e170f39b6ac90e1ff324a90ae88c9a762ec549d0f34"
VAE_SHA = "fc436097e5cbb105c44df4403ae84acf7e81de1f64acfedc334888e9a8eea223"
TEXT_SHA = "8a0859e9385019944df829693ffdef5bbde394950ff561cbfdd61fbb3ea76fb5"


def put(header: bytearray, offset: int, size: int, text: str) -> None:
    encoded = text.encode("ascii") + b"\0"
    if len(encoded) > size:
        raise ValueError("metadata field is too long")
    header[offset : offset + len(encoded)] = encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.snapshot)

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
        prompt = encoder(tokens, return_dict=False)[0].contiguous().float()

    header = bytearray(512)
    header[:8] = b"LOTUSP01"
    struct.pack_into("<IIII", header, 8, 1, 512, 77, 1024)
    put(header, 24, 41, REVISION)
    put(header, 65, 65, UNET_SHA)
    put(header, 130, 65, VAE_SHA)
    put(header, 195, 65, TEXT_SHA)
    Path(args.output).write_bytes(header + prompt.numpy().tobytes())


if __name__ == "__main__":
    main()
