"""Validate all released BigSmall checkpoints with a real CUDA forward pass."""

from collections import OrderedDict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neural_methods.model.BigSmall import BigSmall


def normalize_state_dict(state_dict):
    if state_dict and next(iter(state_dict)).startswith("module."):
        return OrderedDict((name[7:], value) for name, value in state_dict.items())
    return state_dict


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the CUDA 12.8 PyTorch build.")

    device = torch.device("cuda:0")
    checkpoint_dir = REPO_ROOT / "final_model_release"
    checkpoints = sorted(checkpoint_dir.glob("BP4D_BigSmall_Multitask_Fold*.pth"))
    if len(checkpoints) != 3:
        raise FileNotFoundError(f"Expected 3 BigSmall checkpoints, found {len(checkpoints)}")

    torch.manual_seed(100)
    big_input = torch.randn(3, 3, 144, 144, device=device)
    small_input = torch.randn(3, 3, 9, 9, device=device)

    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(device)}")

    for checkpoint in checkpoints:
        model = BigSmall(n_segment=3).to(device).eval()
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(normalize_state_dict(state_dict))
        with torch.inference_mode():
            au, bvp, respiration = model((big_input, small_input))
        torch.cuda.synchronize(device)
        print(
            f"PASS {checkpoint.name}: "
            f"AU={tuple(au.shape)}, BVP={tuple(bvp.shape)}, RESP={tuple(respiration.shape)}"
        )


if __name__ == "__main__":
    main()
