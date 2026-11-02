"""Download and SHA-256 verify eleven US-government HAER head photographs.

Source URLs and accession IDs are declared in ``blender_liberty_reference_studio``.
Originals remain local/ignored, while the editable reference studio packs them.
Any checksum mismatch fails closed; there is no overwrite of an existing image.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from blender_liberty_reference_studio import ALL_VIEWS


EXPECTED_SHA256 = {
    177: "9ec83f6b6fb4d6e241ba242eee39ad1cdfd45eae842baaa79c4efddecb87fcd4",
    178: "ccbd89ffadbe9f8aec95fba9ccb00df58dc02864372fa735ec91db23a64c4fe5",
    179: "259ef8f433cc6cd3becc3c63b7c1a1f70a068a76266d25730509213884a99865",
    180: "70b6613c4230463f1d5024a9d71c2ed88f96ccad600c3a61ca5de1bd55649296",
    181: "c6f478b2646854fd239db1034e1ac303e6ace46f385168dc56a10fc7814ff461",
    236: "472452f78a2eec2cfc83bd36444eb673262d5391a3dcfed568d9104fcf4c6f74",
    237: "e840982323ade17aaaf4f8e77816d0ccfdeadd920573e1f09e8c80f5d51480cb",
    238: "22e48cb23a2e1a2751322affec06737b779f4e2a0628f7db8ace729358a30757",
    239: "3525a7905641f2394faae5e0dbe15d7aed90ddc591b708f911424463d34cec8e",
    240: "e45f7d9d5d1083dd1b5387055b6e04f6f50711380556826b78197f8855b302e3",
    241: "a4862624fcc6b5339a999e33acd39ff276e5c47887ac9ed4dff44d02ec1adcee",
}


def verify(path: Path, accession: int) -> None:
    if not path.is_file():
        raise ValueError(f"missing source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xd8\xff":
            raise ValueError(f"not a JPEG file: {path}")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256[accession]:
        raise ValueError(f"HAER-{accession} does not match the reviewed SHA-256: {path}")


def fetch_all(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in ALL_VIEWS:
        accession = int(view.survey_id.rsplit("-", 1)[-1])
        destination = directory / view.filename
        if not destination.exists():
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=f"haer-{accession}-", suffix=".tmp", delete=False
            ) as temp:
                candidate = Path(temp.name)
                try:
                    req = urllib.request.Request(view.jpeg_url, headers={
                        "User-Agent": "kumiki-blender-haer-study/1.0"})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            temp.write(chunk)
                            if temp.tell() > 15 * 1024 * 1024:
                                raise ValueError(f"HAER-{accession} exceeds reviewed JPEG size")
                    temp.flush()
                    verify(candidate, accession)
                    os.replace(candidate, destination)
                finally:
                    candidate.unlink(missing_ok=True)
        verify(destination, accession)
        paths.append(destination)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Local ignored directory for verified original source JPEGs")
    args = parser.parse_args()
    try:
        paths = fetch_all(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified {len(paths)} distinct authentic Statue of Liberty HAER photos")


if __name__ == "__main__":
    main()
