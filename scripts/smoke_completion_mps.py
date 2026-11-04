"""Actual macOS MPS tensor compute + CPU isosurface compatibility smoke.

The full TripoSR weights are deliberately not fetched in CI. This verifies
the accelerated device selection and CPU-only mesh extraction boundary.
"""

from __future__ import annotations

import platform

from kumiki_blender.reconstruction.completion_generation import (
    _cpu_marching_cubes, choose_torch_device,
)


def main() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("MPS smoke must run on macOS")
    import torch

    if choose_torch_device(torch, "mps") != "mps":
        raise RuntimeError("MPS device not selected")
    # Exercise an actual Metal tensor operation and return its result to CPU.
    matrix = torch.eye(4, device="mps", dtype=torch.float32)
    result = torch.mm(matrix, matrix).cpu()
    if not torch.allclose(result, torch.eye(4)):
        raise RuntimeError("MPS matrix multiplication produced invalid values")

    indices = torch.linspace(-1.0, 1.0, 16, device="mps")
    xx, yy, zz = torch.meshgrid(indices, indices, indices, indexing="ij")
    field = xx * xx + yy * yy + zz * zz - 0.45
    vertices, faces = _cpu_marching_cubes(field, 0.0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(faces) < 4:
        raise RuntimeError("MPS-to-CPU isosurface conversion produced no triangles")
    print(f"Metal MPS compute and CPU marching cubes passed: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    main()
