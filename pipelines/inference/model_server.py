from pathlib import Path
import modal
from pydantic import BaseModel
import zipfile
import requests
import re
from typing import Optional

# ====== CONFIG ======
APP_NAME = "visight-yolo-prod"
BUCKET_NAME = "visight-data-yusufmoola"
MOUNT_PATH = Path("/bucket")                 
SMOKE_DATA_YAML = MOUNT_PATH / "tmp/smoke_v1" / "data.yaml"
SAVE_DIR = MOUNT_PATH / "models" / "smoke_yolov10n"
STATS_DIR = MOUNT_PATH / "stats" / "training" / "smoke_yolov10n"
RUNS_DIR = Path("/root/data/runs")                

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


S3_SECRET = modal.Secret.from_name( 
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
)

# ====== IMAGE ======
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "ultralytics==8.2.68",     # YOLOv8/v10 support
        "opencv-python==4.10.0.84",
        "numpy>=1.24,<2.0",
        "pyyaml>=6.0",
        "onnx>=1.14.0", 
        "fastapi", "boto3"
    ])
)

app = modal.App(APP_NAME, image=image)

def zip_directory(directory_paths, other_paths, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for directory_path in directory_paths:
            for file_path in Path(directory_path).rglob('*'):  # rglob for recursive
                if file_path.is_file():
                    arcname = file_path.relative_to(directory_path)
                    zf.write(file_path, arcname=f"{str(directory_path.stem)}/{arcname}")
        for path in other_paths: zf.write(path)

def download_from_google_drive(share_link, output_path):
    """
    Download a file from a public Google Drive link.
    
    Args:
        share_link: Google Drive share link (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
        output_path: Path where the file will be saved (e.g., 'video.mp4')
    """
    # Extract file ID from the share link
    file_id = extract_file_id(share_link)
    
    if not file_id:
        print("Error: Could not extract file ID from the link")
        return False
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large files that require confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True)
            break
    
    # Download the file
    try:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        
        print(f"\n✓ Downloaded successfully to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def extract_file_id(share_link):
    """Extract file ID from various Google Drive link formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, share_link)
        if match:
            return match.group(1)
    
    return None

class InferenceRequest(BaseModel):
    video_url: str
    fps: Optional[int] = 1
    confidence_threshold: Optional[float] = 0.5

@app.function(
    gpu="t4",
    cpu=2,
    timeout=60 * 10,
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=S3_SECRET)},
    secrets=[S3_SECRET]
)
@modal.fastapi_endpoint(method="POST")
def inference(request: InferenceRequest, save_to_s3: bool = False): 
    import tempfile
    from fastapi.responses import FileResponse
    from pipelines.inference.pipeline_remote import InferencePipeline
    from pipelines.configs.config import Config
    from datetime import datetime
    
    local_vid_path = "sample_vid.mp4"
    
    logger.info("Downloading video to Modal container")
    # vid_url = "https://drive.google.com/file/d/1ya6iuzDMhqCSZG8uRpLsvrNeNA77d8Ew/view?usp=sharing"
    download_from_google_drive(request.video_url, local_vid_path)
    logger.info("Finished downloading video")
    
    model_path = MOUNT_PATH / Path(Config.prod_model_key)
        
    pipeline = InferencePipeline(
        model_path=model_path / "best.pt",
        fps=request.fps,
        confidence_threshold=request.confidence_threshold, 
        logger=logger
    )
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    video_id = f"{str(Path(Config.prod_model_key).stem)}_{timestamp}"
    
    annotated_frame_dir, res_json_path = pipeline.run_inference_on_video(
        video_path=local_vid_path, 
        video_id=video_id, 
        s3_bucket=Config.s3_bucket if save_to_s3 else save_to_s3,
        save_annotated_frames=True
    )
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    logger.info("Started zipping results")  
    try: 
        zip_directory([annotated_frame_dir], [res_json_path], temp_zip.name)
    except Exception as e: 
        raise Exception(f"Error during zip: {e}")
    logger.info("Finished zipping")   
         
    return FileResponse(
        temp_zip.name,
        media_type='application/zip',
        filename='results.zip'
    )
