from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import trimesh
from PIL import Image

from torchtitan.experiments.vem.datasets.bos import BOSClient, COSClient
from torchtitan.experiments.vem.datasets.mesh_utils import Mesh


class DatasetPathIO:
    """Read local, BOS, and COS dataset assets from path strings."""

    REMOTE_SCHEMES = {"bos", "cos"}

    def __init__(self) -> None:
        self._bos_client = None
        self._cos_client = None

    def is_remote(self, path: str) -> bool:
        return urlparse(path).scheme in self.REMOTE_SCHEMES

    def join(self, root: str, *parts: str) -> str:
        parts = [str(p).strip("/") for p in parts if p is not None and str(p) != ""]
        if not parts:
            return root
        first = str(parts[0])
        if urlparse(first).scheme in self.REMOTE_SCHEMES or os.path.isabs(first):
            return first

        parsed = urlparse(root)
        if parsed.scheme in self.REMOTE_SCHEMES:
            base = parsed.path.strip("/")
            key = "/".join([p for p in [base, *parts] if p])
            return f"{parsed.scheme}://{parsed.netloc}/{key}"
        return os.path.join(root, *parts)

    def open_binary(self, path: str):
        parsed = urlparse(path)
        if parsed.scheme == "bos":
            if self._bos_client is None:
                self._bos_client = BOSClient()
            return self._bos_client.get_file(parsed.netloc, parsed.path.lstrip("/"))
        if parsed.scheme == "cos":
            if self._cos_client is None:
                self._cos_client = COSClient()
            return self._cos_client.get_file(parsed.netloc, parsed.path.lstrip("/"))
        return open(path, "rb")

    def read_json(self, path: str):
        with self.open_binary(path) as f:
            if path.endswith(".json.gz"):
                with gzip.open(f, "rt", encoding="UTF-8") as gz:
                    return json.load(gz)
            return json.load(io.TextIOWrapper(f, encoding="UTF-8"))

    def read_parquet(self, path: str) -> pd.DataFrame:
        with self.open_binary(path) as f:
            return pd.read_parquet(f)

    def read_image(self, path: str, mode: str = "RGBA") -> Image.Image:
        with self.open_binary(path) as f:
            image = Image.open(f)
            image.load()
        return image.convert(mode)

    def read_mesh(self, path: str, process: bool = False):
        pc = None
        suffix = os.path.splitext(urlparse(path).path)[1].lower()

        if not self.is_remote(path):
            if suffix == ".npz":
                data = np.load(path)
                vertices = data["v"].astype(np.float32)
                faces = data["f"].astype(np.int64)
                if "p" in data:
                    pc = data["p"].astype(np.float32)
                return trimesh.Trimesh(vertices=vertices, faces=faces, process=process), pc
            if suffix in {".glb", ".obj", ".ply"}:
                return trimesh.load(path, force="mesh", process=process), pc
            raise NotImplementedError(f"Unsupported mesh format: {path}")

        with self.open_binary(path) as f:
            if suffix == ".npz":
                with np.load(f) as data:
                    vertices = data["v"].astype(np.float32)
                    faces = data["f"].astype(np.int64)
                    if "p" in data:
                        pc = data["p"].astype(np.float32)
                return trimesh.Trimesh(vertices=vertices, faces=faces, process=process), pc
            if suffix in {".glb", ".obj", ".ply"}:
                return trimesh.load_mesh(f, file_type=suffix[1:], process=process), pc
            raise NotImplementedError(f"Unsupported mesh format: {path}")

    def read_mixed_mesh(self, path: str) -> Mesh:
        suffix = os.path.splitext(urlparse(path).path)[1].lower()
        if suffix != ".ply":
            raise NotImplementedError(f"Unsupported mixed mesh format: {path}")

        if not self.is_remote(path):
            return Mesh.load(path)

        with self.open_binary(path) as f:
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp.flush()
                return Mesh.load(tmp.name)
