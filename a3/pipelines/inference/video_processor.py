import cv2
import cProfile
import pstats
from io import StringIO
from pathlib import Path
import hashlib
from typing import Optional, List
from datetime import datetime, timezone
import json
import cProfile
import pstats
from io import StringIO
from datetime import datetime, timezone


class VideoProcessor:
    
    def __init__(self, fps: Optional[int] = None):
        self.target_fps = fps
    
    def extract_frames(self, video_path: str, output_dir: str) -> List[dict]:
        """
        Extract frames from video and save to output_dir. 
        
        Returns list of metadata for each frame:
        - frame_id: unique identifier (hash)
        - frame_number: sequential frame number
        - timestamp: timestamp in video (seconds)
        - file_path: path to saved frame image
        
        PROFILED FUNCTION
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.target_fps and self.target_fps < video_fps:
                frame_interval = int(video_fps / self.target_fps)
            else:
                frame_interval = 1
            
            print(f"Video FPS: {video_fps}, Total frames: {total_frames}")
            print(f"Extracting every {frame_interval} frame(s)")
            
            frames_metadata = []
            frame_count = 0
            extracted_count = 0
            
            # Use grab()/retrieve() to avoid decoding frames we will skip
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    break

                if frame_count % frame_interval == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        frame_count += 1
                        continue
                    timestamp = frame_count / video_fps
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = output_path / frame_filename

                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    frame_id = hashlib.sha256(f"{video_path}_{frame_count}".encode()).hexdigest()

                    frames_metadata.append({
                        "frame_id": frame_id,
                        "frame_number": frame_count,
                        "timestamp": round(timestamp, 3),
                        "file_path": str(frame_path)
                    })

                    extracted_count += 1

                frame_count += 1
            
            cap.release()
            
            print(f"Extracted {extracted_count} frames from {total_frames} total frames")
            
            # Save metadata
            metadata_path = output_path / "frames_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "video_path": str(video_path),
                    "video_fps": video_fps,
                    "total_frames": total_frames,
                    "extracted_frames": extracted_count,
                    "frame_interval": frame_interval,
                    "frames": frames_metadata
                }, f, indent=2)
            
            return frames_metadata
        finally:
            profiler.disable()
            
            # Save profiling stats
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            
            # Save profile to output directory
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            profile_output_dir = Path(output_dir).parent / "profiling"
            profile_output_dir.mkdir(parents=True, exist_ok=True)
            profile_file = profile_output_dir / f"extract_frames_{timestamp}.txt"
            profile_file.write_text(s.getvalue())
            print(f"\n[PROFILING] extract_frames profile saved to: {profile_file}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video file
        
        Returns:
            Dict with video metadata (fps, duration, resolution, etc.)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 2),
            "resolution": f"{width}x{height}"
        }
