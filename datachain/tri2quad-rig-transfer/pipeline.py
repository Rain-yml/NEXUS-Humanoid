from __future__ import annotations

from artifact import encode_artifact
from blender_rig import normalize_source
from bos_io import BOS, split_bos_uri
from contract import Rejection, SCHEMA
from correspondence import transfer_rig
from mesh_io import read_ply
from scipy.spatial import cKDTree

from contract import MAX_COORDINATE_ERROR


def artifact_key(prefix: str, uuid: str) -> str:
    root = prefix.strip("/")
    return f"{root}/{uuid[:2]}/{uuid}.npz" if root else f"{uuid[:2]}/{uuid}.npz"


def _half_turn_symmetric(vertices) -> bool:
    rotated = vertices * [-1.0, -1.0, 1.0]
    distances, indices = cKDTree(vertices).query(rotated)
    return len(set(indices.tolist())) == len(vertices) and float(distances.max(initial=0.0)) <= MAX_COORDINATE_ERROR


def solve(source: bytes, final):
    failures = []
    try:
        normalized = normalize_source(source, plus_y_front=False)
        transfer = transfer_rig(
            normalized.rig, *final, normalized.source_to_normalized
        )
        if _half_turn_symmetric(transfer.final_vertices):
            raise Rejection(
                "ambiguous_producer_mode",
                "final geometry is exactly symmetric under the producer's 180-degree front rotation",
            )
        return normalized, transfer
    except Rejection as error:
        if error.code == "ambiguous_producer_mode":
            raise
        failures.append(f"native_front={error.code}: {error}")
    try:
        normalized = normalize_source(source, plus_y_front=True)
        transfer = transfer_rig(
            normalized.rig, *final, normalized.source_to_normalized
        )
        return normalized, transfer
    except Rejection as error:
        failures.append(f"plus_y_front={error.code}: {error}")
    raise Rejection("producer_modes_failed", " | ".join(failures))


def process(row: dict, bos: BOS, output_bucket: str, output_prefix: str) -> dict:
    uuid = str(row["uuid"])
    source = bos.read(str(row["bos_bucket"]), str(row["bos_uri"]))
    final_bucket, final_key = split_bos_uri(str(row["mesh_path"]))
    final_vertices, final_triangles, final_quads = read_ply(
        bos.read(final_bucket, final_key)
    )
    normalized, transfer = solve(
        source, (final_vertices, final_triangles, final_quads)
    )
    output_key = artifact_key(output_prefix, uuid)
    payload = encode_artifact(uuid, transfer)
    bos.write(output_bucket, output_key, payload)
    return {
        "uuid": uuid,
        "status": "accepted",
        "reason": "",
        "source": str(row.get("source", "")),
        "source_bos_bucket": str(row["bos_bucket"]),
        "source_bos_uri": str(row["bos_uri"]),
        "final_mesh_uri": str(row["mesh_path"]),
        "output_bos_bucket": output_bucket,
        "output_bos_key": output_key,
        "vertex_count": len(transfer.final_vertices),
        "joint_count": len(transfer.joint_names),
        "weight_width": transfer.weight_indices.shape[1],
        "artifact_bytes": len(payload),
        "max_coordinate_error": transfer.max_coordinate_error,
        "producer_mode": normalized.producer_mode,
        "schema": SCHEMA,
    }


def rejection(row: dict, error: Exception) -> dict:
    return {
        "uuid": str(row["uuid"]),
        "status": "rejected",
        "reason": error.code if isinstance(error, Rejection) else "unexpected_error",
        "detail": str(error),
        "source": str(row.get("source", "")),
        "source_bos_bucket": str(row.get("bos_bucket", "")),
        "source_bos_uri": str(row.get("bos_uri", "")),
        "final_mesh_uri": str(row.get("mesh_path", "")),
        "output_bos_bucket": "",
        "output_bos_key": "",
        "vertex_count": 0,
        "joint_count": 0,
        "weight_width": 0,
        "artifact_bytes": 0,
        "max_coordinate_error": None,
        "producer_mode": "",
        "schema": SCHEMA,
    }
