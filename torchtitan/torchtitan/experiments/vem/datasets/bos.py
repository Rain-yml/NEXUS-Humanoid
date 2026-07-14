import os
import io

from baidubce.services.bos.bos_client import BosClient
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration


class BOSClient:
    def __init__(self):
        bos_endpoint = os.getenv("BOS_ENDPOINT")
        access_key_id = os.getenv("BOS_ACCESS_KEY")
        secret_access_key = os.getenv("BOS_SECRET_KEY")
        config = BceClientConfiguration(
            credentials=BceCredentials(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key
            ),
            endpoint=bos_endpoint
        )

        client = BosClient(config)
        self.client = client
    
    def get_file(self, bos_bucket: str, bos_path: str):
        try:
            return io.BytesIO(self.client.get_object_as_string(bos_bucket, bos_path))
        except Exception as e:
            raise RuntimeError(f"Failed to get file {bos_bucket}/{bos_path} from BOS.")

    def put_file(self, bos_bucket: str, bos_path: str, file: io.BytesIO):
        self.client.put_object_from_string(bos_bucket, bos_path, file.getvalue())

    def put_local_file(self, bos_bucket: str, bos_path: str, file_path: str):
        self.client.put_object_from_file(bos_bucket, bos_path, file_path)

class COSClient:
    def __init__(self):
        from qcloud_cos import CosConfig
        from qcloud_cos import CosS3Client

        import logging

        cos_logger = logging.getLogger("qcloud_cos.cos_client")
        cos_logger.setLevel(logging.ERROR)

        region = os.getenv("COS_REGION")
        secret_id = os.getenv("COS_SECRET_ID")
        secret_key = os.getenv("COS_SECRET_KEY")
        config = CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Token=None,
            Scheme='https',
        )
        self.client = CosS3Client(config)
    
    def get_file(self, bucket_name, object_key):
        try:
            response = self.client.get_object(bucket_name, object_key)
            return io.BytesIO(response['Body'].get_raw_stream().read())
        except Exception as e:
            raise RuntimeError(f"Failed to get file {bucket_name}/{object_key} from COS.")