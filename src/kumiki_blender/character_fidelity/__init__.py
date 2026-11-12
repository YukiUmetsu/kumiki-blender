"""Read-only character-sheet / Blender semantic fidelity QA.

Evaluation dependencies are intentionally lazy: Blender's bundled Python does not
need OpenCV, SciPy or Pillow to capture geometry, color and depth passes.
"""

__all__ = ["load_manifest", "evaluate_character"]


def __getattr__(name):
    if name == "load_manifest":
        from .schema import load_manifest
        return load_manifest
    if name == "evaluate_character":
        from .evaluate import evaluate_character
        return evaluate_character
    raise AttributeError(name)
