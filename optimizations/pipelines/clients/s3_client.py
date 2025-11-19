import boto3
import yaml
from pathlib import Path
import re
from typing import Optional
from botocore.exceptions import ClientError


class DataClient: 
    
    def get_object(self):
        raise NotImplementedError("Expected concrete implementation of get_object in DataClient class")
        
    def supported_formats(self):
        return {
            "yaml": self.parse_yaml,
            "txt" : self.parse_txt,
        }
    
    def _warn_filename(self, filename, filetype): 
        if Path(filename).suffix != f".{filetype}": 
            print(f"WARNING: attempting to parse file {filename} as .{filetype}")
    
    def parse_txt(self, filename, byte_data):
        self._warn_filename(filename, "txt")
            
        try: 
            return byte_data.decode("utf-8")
        except Exception as exc:
            raise Exception(f"Failed to parse file {filename} as txt: {exc}")
    
    def parse_yaml(self, filename, byte_data):
        self._warn_filename(filename, "yaml")
            
        try: 
            data = yaml.safe_load(byte_data)
        except yaml.YAMLError as exc:
            raise Exception(f"Failed to parse file {filename} as yaml: {exc}")
        
        return data
        

class s3Client(DataClient):
    
    def __init__(self, buckets): 
        self.client = boto3.client('s3')
        exceptions={}
        valid_found = False
        for bucket in buckets: 
            try: 
                self.bucket = bucket
                self.client.head_bucket(Bucket=bucket)
                valid_found = True
                break
            except Exception as e: exceptions[bucket] = e
            
        if not valid_found: raise Exception("All bucket options failed! Error messages: ", exceptions)
        
        try: 
            response = self.client.get_bucket_versioning(
                Bucket=self.bucket,
            )
            self.versioning_enabled = response.get("Status")
        except: self.versioning_enabled = "Unknown"            
            
    def _check_and_fix_key(self, key, max_version_lookback=10): 
        try: 
            res = self.client.list_object_versions(Bucket=self.bucket, Prefix=key, MaxKeys=max_version_lookback)
            for v in res.get("Versions"): 
                response = self.client.get_object(Bucket=self.bucket, Key=key, VersionId=v.get("VersionId"))["Body"]
                stream = response.read()
                return stream
        except Exception as e: 
            raise Exception(f"Failed to fetch {self.bucket}/{key} from s3: {e}")
        
    def _add_bucket_prefix(self, filepath):
        return f"{self.bucket}/{filepath}"
        
    def _remove_bucket_prefix(self, full_filepath): 
        return re.sub(fr"^{self.bucket}/", "", full_filepath)
    
    def get_object(self, key, parse_format:Optional[str]=None, auto_check_prev_versions:bool=False):        
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            stream = response.read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey' and auto_check_prev_versions:
                stream = self._check_and_fix_key(key)
                # TODO: add log indicating that previous version was used
            else: raise Exception(f"Failed to fetch {self.bucket}/{key} from s3: {e}")

        parse_fxn = self.supported_formats().get(parse_format) if parse_format else None
        if parse_fxn: 
            return parse_fxn(filename=key, byte_data=stream)
        else:
            # This means the file format is not supported 
            print(f"WARNING: Parse type unprovided or unsupported by local s3Client class, returning object in bytes.")
            return stream
    
    def batch_get_filenames(self, prefix, verbose=False):
        file_paths = []
        
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    cur_page_file_paths = [self._add_bucket_prefix(obj["Key"]) for obj in page["Contents"] 
                                           if obj["Key"] != f"{prefix}/"]
                    file_paths.extend(cur_page_file_paths)
        except Exception as e:
            raise Exception(f"Error while pagenated list_objects_v2: {e}")
        
        if file_paths and verbose: 
            print(f"Retrieved {len(file_paths)} files from: {self.bucket}/{prefix}")
        elif not file_paths: 
            print(f"WARNING: no files exist in {self.bucket}/{prefix}")
        return file_paths

    def put_object(self, key, data, content_type=None):
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                
            if isinstance(data, bytes):
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    **extra_args
                )
            else:
                self.client.upload_fileobj(data, self.bucket, key, ExtraArgs=extra_args)
            
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            raise Exception(f"Failed to upload to {self.bucket}/{key}: {e}")

    def upload_file(self, local_path, key, content_type=None):
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                
            self.client.upload_file(str(local_path), self.bucket, key, ExtraArgs=extra_args)
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            raise Exception(f"Failed to upload file {local_path} to {self.bucket}/{key}: {e}")

    def download_file(self, key, local_path):
        try:
            self.client.download_file(self.bucket, key, str(local_path))
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download {self.bucket}/{key} to {local_path}: {e}")
        
        