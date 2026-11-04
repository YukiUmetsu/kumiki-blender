"""macOS MPS probe with explicit skip when a hosted VM lacks Metal hardware.

The full TripoSR weights are deliberately not fetched in CI. On a real Mac,
--require-mps verifies actual accelerated tensors and CPU mesh extraction.
"""

from __future__ import annotations

import platform
import argparse

from kumiki_blender.reconstruction.completion_generation import (
    _cpu_marching_cubes, choose_torch_device,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-mps", action="store_true",
                        help="Fail if real Apple Metal execution cannot be tested")
    args = parser.parse_args(argv)
    if platform.system() != "Darwin":
        raise RuntimeError("MPS smoke must run on macOS")
    import torch

    mps_available = torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("Apple MPS device is not available on this host")
    if mps_available and choose_torch_device(torch, "mps") != "mps":
        raise RuntimeError("MPS device not selected")
    device = "mps" if mps_available else "cpu"
    if not mps_available:
        print("SKIPPED Metal acceleration: GitHub macOS VM exposes no MPS device; CPU geometry compatibility only")
    # Exercise actual Metal tensor math when exposed and return to CPU.
    matrix = torch.eye(4, device=device, dtype=torch.float32)
    result = torch.mm(matrix, matrix).cpu()
    if not torch.allclose(result, torch.eye(4)):
        raise RuntimeError("MPS matrix multiplication produced invalid values")

    indices = torch.linspace(-1.0, 1.0, 16, device=device)
    xx, yy, zz = torch.meshgrid(indices, indices, indices, indexing="ij")
    field = xx * xx + yy * yy + zz * zz - 0.45
    vertices, faces = _cpu_marching_cubes(field, 0.0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(faces) < 4:
        raise RuntimeError("MPS-to-CPU isosurface conversion produced no triangles")
    print(f"{device} tensor compute and CPU marching cubes passed: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    main()
