"""Generate deterministic Python CPU fixtures for the native Lotus port."""

from __future__ import annotations

import argparse
import enum
import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import torch


# The installed TorchVision wheel may lack its optional compiled operators.
# Diffusers only needs these symbols for unrelated image/video helpers while
# this fixture generator works directly with tensors.
_torchvision_modules = {
    name: types.ModuleType(name)
    for name in (
        "torchvision",
        "torchvision.transforms",
        "torchvision.io",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2.functional",
    )
}
for _name, _module in _torchvision_modules.items():
    _module.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
    sys.modules[_name] = _module


class _InterpolationMode(enum.Enum):
    NEAREST = 0
    NEAREST_EXACT = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
    LANCZOS = 6


_torchvision_modules["torchvision.transforms"].InterpolationMode = (
    _InterpolationMode
)
_torchvision_modules["torchvision"].transforms = _torchvision_modules[
    "torchvision.transforms"
]
_torchvision_modules["torchvision"].io = _torchvision_modules[
    "torchvision.io"
]
_torchvision_modules["torchvision.transforms"].v2 = _torchvision_modules[
    "torchvision.transforms.v2"
]
_torchvision_modules["torchvision.transforms.v2"].functional = (
    _torchvision_modules["torchvision.transforms.v2.functional"]
)

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def save(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().float().numpy()
    path.write_bytes(array.tobytes())
    print(path.name, tuple(array.shape), float(array.min()), float(array.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("output")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.snapshot)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(17)

    size = args.size
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    rgb = torch.stack(
        (
            torch.sin(xx * 0.071),
            torch.cos(yy * 0.053),
            torch.sin((xx + yy) * 0.037),
        ),
        dim=0,
    ).unsqueeze(0)

    tokenizer = CLIPTokenizer.from_pretrained(
        root, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        root, subfolder="text_encoder", local_files_only=True
    ).eval()
    tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    prompt = text_encoder(tokens, return_dict=False)[0]
    del text_encoder

    vae = AutoencoderKL.from_pretrained(
        root, subfolder="vae", local_files_only=True
    ).eval()
    posterior = vae.encode(rgb).latent_dist
    posterior_noise = torch.randn_like(posterior.mean)
    rgb_latent = (
        posterior.mean + posterior.std * posterior_noise
    ) * vae.config.scaling_factor

    initial_latent = torch.randn(
        (1, 4, size // 8, size // 8), dtype=torch.float32
    )
    task = torch.tensor([[1.0, 0.0]])
    task = torch.cat((torch.sin(task), torch.cos(task)), dim=-1)

    unet = UNet2DConditionModel.from_pretrained(
        root, subfolder="unet", local_files_only=True
    ).eval()
    if unet.config.in_channels == 8:
        unet_input = torch.cat((rgb_latent, initial_latent), dim=1)
    elif unet.config.in_channels == 4:
        unet_input = rgb_latent
    else:
        raise RuntimeError(
            f"unsupported Lotus UNet input channels: {unet.config.in_channels}"
        )
    prediction = unet(
        unet_input,
        torch.tensor(999),
        encoder_hidden_states=prompt,
        class_labels=task,
        return_dict=False,
    )[0]
    decoded = vae.decode(
        prediction / vae.config.scaling_factor,
        return_dict=False,
    )[0]
    depth = (decoded * 0.5 + 0.5).clamp(0.0, 1.0).mean(dim=1)

    save(output / "rgb.bin", rgb)
    save(output / "prompt.bin", prompt)
    save(output / "posterior_mean.bin", posterior.mean)
    save(output / "posterior_logvar.bin", posterior.logvar)
    save(output / "posterior_noise.bin", posterior_noise)
    save(output / "rgb_latent.bin", rgb_latent)
    save(output / "initial_latent.bin", initial_latent)
    save(output / "task.bin", task)
    save(output / "unet_prediction.bin", prediction)
    save(output / "decoded.bin", decoded)
    save(output / "depth.bin", depth)


if __name__ == "__main__":
    main()
