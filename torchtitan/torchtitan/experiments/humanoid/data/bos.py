"""Small BOS reader with no dependency on NEXUS train-spec registration."""

from __future__ import annotations

import io
import os


class BOSClient:
    def __init__(self):
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient

        config = BceClientConfiguration(
            credentials=BceCredentials(
                access_key_id=os.environ["BOS_ACCESS_KEY"],
                secret_access_key=os.environ["BOS_SECRET_KEY"],
            ),
            endpoint=os.environ["BOS_ENDPOINT"],
        )
        self.client = BosClient(config)

    def get_file(self, bucket: str, key: str) -> io.BytesIO:
        try:
            return io.BytesIO(self.client.get_object_as_string(bucket, key))
        except Exception as error:
            raise RuntimeError(f"Failed to read bos://{bucket}/{key}") from error
