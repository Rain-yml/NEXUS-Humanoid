import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "humanoid" / "build_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("build_humanoid_manifest", SCRIPT)
build_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_manifest)


def test_artifact_row_uses_explicit_dataset_prefixes():
    row = build_manifest.artifact_row(
        {"uuid": "abcdef", "has_texture": True},
        dataset_prefix="datasets/example",
        render_prefix="renders/example",
        rig_subdir="rig_npz",
        mesh_subdir="glb",
        joint_schema="humanoid-28-v1",
        train_fraction=0.98,
        val_fraction=0.01,
        split="test",
    )

    assert row["split"] == "test"
    assert row["rig_npz_uri"] == (
        "bos://biped-data-resave-glb-npz/datasets/example/rig_npz/ab/abcdef.npz"
    )
    assert row["mesh_glb_uri"] == (
        "bos://biped-data-resave-glb-npz/datasets/example/glb/ab/abcdef.glb"
    )
    assert row["color_view_0_uri"] == (
        "bos://biped-data-render-rgb-normal-4v/renders/example/ab/abcdef/"
        "color_0000.webp"
    )
    assert row["normal_view_3_uri"] == (
        "bos://biped-data-render-rgb-normal-4v/renders/example/ab/abcdef/"
        "normal_0003.webp"
    )


def test_split_for_uuid_is_stable():
    assert build_manifest.split_for_uuid("asset", 0.8, 0.1) == "val"
