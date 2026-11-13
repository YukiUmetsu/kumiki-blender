"""Hunyuan3D-2mv shape from front/left/back/right RGBA views on Apple Silicon (MPS).

    HUNYUAN3D_ROOT=/path/to/Hunyuan3D-2 python scripts/hunyuan_mv_shape.py OUT.glb \
        --front handoff/raw/front.png --left handoff/raw/left.png \
        --back handoff/raw/back.png --right handoff/raw/right.png

Run it in the Hunyuan3D-2 virtual environment (transformers 4.48, diffusers 0.32: newer
versions rename the DINOv2 weight keys). Hunyuan3D-2mv's "left" is the character's left,
the same as this repo's view names. Diffusion runs on MPS in fp16; decoding runs on MPS in
fp32, because the fp16 decode hung the Metal queue. The latents are saved next to the GLB.
Check the Tencent Hunyuan 3D 2.0 license (territory and user limits) before use.
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image
sys.path.insert(0, os.environ.get("HUNYUAN3D_ROOT", "Hunyuan3D-2"))
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover

p = argparse.ArgumentParser()
p.add_argument("out")
for v in ("front", "left", "back", "right"):
    p.add_argument(f"--{v}")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--octree", type=int, default=380)
p.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv")
a = p.parse_args()
images = {v: Image.open(getattr(a, v)).convert("RGBA") for v in ("front", "left", "back", "right") if getattr(a, v)}
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mv", subfolder=a.subfolder,
                                                        variant="fp16", device="mps", dtype=torch.float16)
t = time.time()
# Diffusion on MPS in fp16, decoding on MPS in fp32: the fp16 decode hung the Metal queue.
latents = pipe(image=images, num_inference_steps=a.steps, octree_resolution=a.octree, num_chunks=20000,
               generator=torch.manual_seed(a.seed), output_type="latent")
latents = latents.detach().to("cpu", torch.float32)
torch.save(latents, a.out + ".latents.pt")  # decode again later at another resolution without resampling
pipe.vae.to("mps", torch.float32)
latents = latents.to("mps")
from hy3dgen.shapegen.pipelines import export_to_trimesh
with torch.no_grad():
    mesh = export_to_trimesh(pipe.vae.latents2mesh(pipe.vae(latents / pipe.vae.scale_factor), bounds=1.01,
                                                   mc_level=0.0, num_chunks=20000, octree_resolution=a.octree,
                                                   mc_algo="mc"))[0]
if mesh is None:
    sys.exit("empty surface: no mesh")
mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
mesh.export(a.out)
print(f"done {time.time() - t:.0f}s faces {len(mesh.faces)}")
