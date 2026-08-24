from __future__ import annotations

import os
from pathlib import Path

from rigviz import ViewerConfig, serve

from source import TransferResultSource


def main() -> None:
    source = TransferResultSource(Path(os.environ["RESULT_PARTS_DIR"]))
    serve(
        source,
        config=ViewerConfig(
            title="Tri2Quad Rig Transfer",
            eyebrow="STRICT ACCEPTED ASSETS",
        ),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8769")),
    )


if __name__ == "__main__":
    main()
