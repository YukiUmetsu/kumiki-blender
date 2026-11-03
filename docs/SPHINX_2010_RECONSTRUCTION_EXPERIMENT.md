# Great Sphinx: same-day Commons photo reconstruction experiment

This experiment uses five **real photographs of the Great Sphinx of Giza**
taken by [Wknight94 on 6 February 2010](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010.jpg).
It **does not** use the rejected procedural Sphinx model.

## Inputs

- [Great Sphinx 2010](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010.jpg)
- [Great Sphinx 2010 2](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010_2.jpg)
- [Great Sphinx 2010 4](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010_4.jpg)
- [Great Sphinx 2010 5](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010_5.jpg)
- [Great Sphinx 2010 rear](https://commons.wikimedia.org/wiki/File:Great_Sphinx_2010_rear.jpg)

These are **only five photos**, not a guaranteed overlapping photo sequence.
They have shared photographer/date/camera provenance, but their camera positions,
subject framing, and geometric overlap must still be verified.

The photos are available under **CC BY-SA 3.0** (or GFDL).
The downloader records source URLs, author, license, original SHA256, and
dimensions. Original photos remain private and are not committed to GitHub.

## Run on a Mac with PyCOLMAP

```bash
python3 -m pip install -e '.[photogrammetry]' 'trimesh>=4,<5'
python3 scripts/experiment_sphinx_commons_2010.py \
  --output ".review/sphinx-2010-$(date -u +%Y%m%dT%H%M%SZ)" \
  --profile high-recall --max-image-size 3200
```

If Wikimedia responds with HTTP 429 (rate limit), **do not retry aggressively**.
Use the five source pages above to download their original JPEGs in your browser
into a local folder, preserving these exact filenames:

```text
Great_Sphinx_2010.jpg
Great_Sphinx_2010_2.jpg
Great_Sphinx_2010_4.jpg
Great_Sphinx_2010_5.jpg
Great_Sphinx_2010_rear.jpg
```

Then run the same command with
`--offline-photos /absolute/path/to/downloaded-jpegs`.
This avoids repeated automated requests to Wikimedia.

## Evidence and limits

The script saves:

- `result.json` and `source-metadata.json`: immutable experiment results and provenance.
- `sparse-reconstruction/reconstruction-report.json`: actual PyCOLMAP stage results.
- `registration-graph.json` and `registration-contact-sheet.jpg`: matches and camera groups.
- `sparse-points.ply` and `sparse-points.glb`, **only if points triangulate**.

The GLB, if produced, is a **colored 3D point cloud**, not a watertight surface
mesh or a verified reconstruction of the monument's facial anatomy.
There is **no procedural fallback**. A surface mesh requires adequate registered
views and validated dense reconstruction. The production OpenSplat quality gate
(8+ registered cameras and 60%+ independent-image registration) is unchanged.

### GitHub Actions result

The initial hosted-runner attempt
[failed at image download](https://github.com/YukiUmetsu/kumiki-blender/actions/runs/41581718992):
Wikimedia returned **HTTP 429 Too Many Requests** for the first original photo.
No cameras were registered and **no 3D reconstruction was generated** in that run.
The Mac/browser download path above is the next legitimate test.
