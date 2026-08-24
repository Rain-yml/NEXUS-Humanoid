from __future__ import annotations

import io
import os

from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient


class BOS:
    def __init__(self):
        self.client = BosClient(
            BceClientConfiguration(
                credentials=BceCredentials(
                    os.environ["BOS_ACCESS_KEY"], os.environ["BOS_SECRET_KEY"]
                ),
                endpoint=os.environ["BOS_ENDPOINT"],
            )
        )

    def read(self, bucket: str, key: str) -> bytes:
        return self.client.get_object_as_string(bucket, key)

    def write(self, bucket: str, key: str, payload: bytes) -> None:
        self.client.put_object_from_string(bucket, key, payload)


def split_bos_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("bos://"):
        raise ValueError(f"expected a bos:// URI, got {uri!r}")
    bucket, separator, key = uri[6:].partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"invalid BOS URI: {uri!r}")
    return bucket, key
