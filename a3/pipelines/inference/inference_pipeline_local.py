import argparse
import cProfile
import pstats
from io import StringIO
from pathlib import Path
import json
import tempfile
import cv2
from datetime import datetime, timezone
from typing import Optional, List
from pipelines.inference.video_processor import VideoProcessor
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from pipelines.clients.s3_client import s3Client


class InferencePipeline:
    """
    Main inference pipeline for video logo detection
    """
    
    def __init__(self, s3_bucket: str, model_key: str, fps: int = None, confidence_threshold: float = 0.5, batch_size: Optional[int] = None):
        self.s3_client = s3Client(buckets=[s3_bucket])
        self.video_processor = VideoProcessor(fps=fps)
        self.bucket = s3_bucket
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        pt_weights = self.s3_client.get_object(model_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(pt_weights)
            tmp_path = tmp_file.name

        self.model = YOLO(tmp_path)    
    
    def run_inference_on_video(self, video_path: str, video_id: str, 
                               upload_to_s3: bool = True, 
                               save_annotated_frames: bool = True) -> dict:
        """
        Complete inference pipeline for a single video
        
        Args:
            video_path: local path to video file
            video_id: unique identifier for this video
            upload_to_s3: whether to upload frames to S3
            save_annotated_frames: whether to save frames with bounding boxes drawn
            
        Returns:
            Dict with pipeline results including frame metadata and inference results
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Get video info
        print("Step 1: Extracting video metadata...")
        video_info = self.video_processor.get_video_info(video_path)
        print(f"Video info: {video_info}")
        
        # Step 2: Upload original video to S3
        if upload_to_s3:
            print("\nStep 2: Uploading video to S3...")
            video_key = f"inference/videos/{video_id}.mp4"
            video_s3_path = self.s3_client.upload_file(
                video_path, 
                video_key, 
                content_type='video/mp4'
            )
            print(f"Video uploaded to: {video_s3_path}")
        else:
            video_s3_path = None
        
        # Step 3: Extract frames
        print("\nStep 3: Extracting frames...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_dir = Path(tmp_dir) / "frames"
            annotated_dir = Path(tmp_dir) / "annotated" if save_annotated_frames else None
            if annotated_dir:
                annotated_dir.mkdir(parents=True, exist_ok=True)
            
            frames_metadata = self.video_processor.extract_frames(video_path, str(frames_dir))
            
            # Step 4: Upload original frames to S3
            if upload_to_s3:
                print("\nStep 4: Uploading original frames to S3...")
                s3_frames = []
                for frame_meta in tqdm(frames_metadata, desc="Uploading frames"):
                    frame_path = Path(frame_meta["file_path"])
                    frame_key = f"inference/frames/{video_id}/original/{frame_path.name}"
                    
                    frame_s3_path = self.s3_client.upload_file(
                        frame_path,
                        frame_key,
                        content_type='image/jpeg'
                    )
                    
                    s3_frames.append({
                        **frame_meta,
                        "s3_path": frame_s3_path
                    })
                
                print(f"Uploaded {len(s3_frames)} frames to S3")
            else:
                s3_frames = frames_metadata
            
            # Step 5: Run inference on frames
            print("\nStep 5: Running inference on frames...")
            inference_results = self._run_model_inference(
                frames_metadata, 
                frames_dir, 
                annotated_dir
            )
            
            # Step 6: Upload annotated frames to S3
            if upload_to_s3 and save_annotated_frames and annotated_dir:
                print("\nStep 6: Uploading annotated frames to S3...")
                for result in tqdm(inference_results, desc="Uploading annotated frames"):
                    if result.get("annotated_frame_path"):
                        annotated_path = Path(result["annotated_frame_path"])
                        annotated_key = f"inference/frames/{video_id}/annotated/{annotated_path.name}"
                        
                        annotated_s3_path = self.s3_client.upload_file(
                            annotated_path,
                            annotated_key,
                            content_type='image/jpeg'
                        )
                        result["annotated_s3_path"] = annotated_s3_path
                
                print(f"Uploaded {len(inference_results)} annotated frames to S3")
            
            # Step 7: Generate summary statistics
            print("\nStep 7: Generating summary statistics...")
            summary_stats = self._generate_summary_stats(inference_results)
            
            # Step 8: Aggregate results
            print("\nStep 8: Aggregating results...")
            pipeline_results = {
                "video_id": video_id,
                "video_info": video_info,
                "video_s3_path": video_s3_path,
                "confidence_threshold": self.confidence_threshold,
                "total_frames": len(frames_metadata),
                "frames": s3_frames if upload_to_s3 else frames_metadata,
                "inference_results": inference_results,
                "summary_stats": summary_stats
            }
            
            # Save results
            if upload_to_s3:
                results_key = f"inference/results/{video_id}/results.json"
                results_bytes = json.dumps(pipeline_results, indent=2).encode('utf-8')
                results_s3_path = self.s3_client.put_object(
                    results_key,
                    results_bytes,
                    content_type='application/json'
                )
                print(f"\nResults saved to: {results_s3_path}")
                pipeline_results["results_s3_path"] = results_s3_path
        
        print(f"\n{'='*60}")
        print(f"Pipeline complete for video: {video_id}")
        print(f"Total detections: {summary_stats['total_detections']}")
        print(f"Frames with detections: {summary_stats['frames_with_detections']}, out of {summary_stats['total_frames']} frames")
        print(f"{'='*60}\n")
        
        return pipeline_results
    
    def _run_model_inference(self, frames_metadata: list, frames_dir: Path, 
                            annotated_dir: Path = None) -> list:
        """
        Run model inference on extracted frames
        
        Args:
            frames_metadata: list of frame metadata dicts
            frames_dir: directory containing frame images
            annotated_dir: directory to save annotated frames (optional)
            
        Returns:
            List of inference results per frame with structure:
            {
                "frame_id": str,
                "frame_number": int,
                "timestamp": float,
                "detections": [
                    {
                        "class_id": int,
                        "class_name": str,
                        "confidence": float,
                        "bbox": [x1, y1, x2, y2]  # coordinates in pixels
                    }
                ],
                "detection_count": int,
                "annotated_frame_path": str (if annotated_dir provided)
            }
            
        PROFILED FUNCTION
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            print("Running model inference...")
            results = []

            # Use batched prediction when batch_size > 1
            effective_bs = self.batch_size if (self.batch_size and self.batch_size > 1) else None

            if not effective_bs:
                for frame_meta in tqdm(frames_metadata, desc="Inference"):
                    frame_path = Path(frame_meta["file_path"])
                    frame_detections = next(self.model.predict([frame_meta["file_path"]],
                                                               conf=self.confidence_threshold, verbose=False, stream=True))
                    name_map = frame_detections.names
                    detection_data = frame_detections.boxes.data.tolist()
                    detection_info = [{"bbox": d[:4], "confidence": d[4], "class_name": name_map[d[5]]} for d in detection_data]
                    result = {
                        "frame_id": frame_meta["frame_id"],
                        "frame_number": frame_meta["frame_number"],
                        "timestamp": frame_meta["timestamp"],
                        "detections": detection_info,
                        "detection_count": len(detection_info)
                    }
                    if annotated_dir and len(detection_info) > 0:
                        annotated_path = self._draw_bounding_boxes(frame_path, detection_info, annotated_dir)
                        result["annotated_frame_path"] = str(annotated_path)
                    results.append(result)
                return results

            # Batched path
            def chunk(items: List[dict], n: int):
                for i in range(0, len(items), n):
                    yield items[i:i+n]

            for batch in tqdm(list(chunk(frames_metadata, effective_bs)), desc=f"Inference (batch={effective_bs})"):
                paths = [f["file_path"] for f in batch]
                all_dets = self.model.predict(paths, conf=self.confidence_threshold, verbose=False, stream=True)
                i = 0
                for det in all_dets:
                    name_map = det.names
                    detection_data = det.boxes.data.tolist()
                    detection_info = [{"bbox": d[:4], "confidence": d[4], "class_name": name_map[d[5]]} for d in detection_data]
                    meta = batch[i]
                    result = {
                        "frame_id": meta["frame_id"],
                        "frame_number": meta["frame_number"],
                        "timestamp": meta["timestamp"],
                        "detections": detection_info,
                        "detection_count": len(detection_info)
                    }
                    if annotated_dir and len(detection_info) > 0:
                        annotated_path = self._draw_bounding_boxes(Path(meta["file_path"]), detection_info, annotated_dir)
                        result["annotated_frame_path"] = str(annotated_path)
                    results.append(result)
                    i += 1
            return results
        finally:
            profiler.disable()
            
            # Save profiling stats
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            
            # Save profile to frames directory parent
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            profile_output_dir = frames_dir.parent / "profiling"
            profile_output_dir.mkdir(parents=True, exist_ok=True)
            profile_file = profile_output_dir / f"run_model_inference_{timestamp}.txt"
            profile_file.write_text(s.getvalue())
            print(f"\n[PROFILING] _run_model_inference profile saved to: {profile_file}")
    
    def _draw_bounding_boxes(self, frame_path: Path, detections: list, 
                            output_dir: Path) -> Path:
        """
        Draw bounding boxes on frame and save annotated image
        
        Args:
            frame_path: path to original frame
            detections: list of detection dicts with bbox and class info
            output_dir: directory to save annotated frame
            
        Returns:
            Path to annotated frame
        """
        # Read image
        img = cv2.imread(str(frame_path))
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Draw bounding box (green)
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw filled rectangle for text background
            cv2.rectangle(img, 
                         (int(x1), int(y1) - text_height - 10), 
                         (int(x1) + text_width, int(y1)), 
                         color, -1)
            
            # Draw text
            cv2.putText(img, label, 
                       (int(x1), int(y1) - 5), 
                       font, font_scale, (0, 0, 0), font_thickness)
        
        # Save annotated frame
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), img)
        
        return output_path
    
    def _generate_summary_stats(self, inference_results: list) -> dict:
        """
        Generate summary statistics from inference results
        
        Args:
            inference_results: list of per-frame inference results
            
        Returns:
            Dict with summary statistics
        """
        total_detections = 0
        frames_with_detections = 0
        class_counts = {}
        class_conf_sum = {}
        class_conf_n = {}

        for result in inference_results:
            n = result.get("detection_count", 0)
            total_detections += n
            if n > 0:
                frames_with_detections += 1
            for det in result.get("detections", []):
                cls = det["class_name"]
                class_counts[cls] = class_counts.get(cls, 0) + 1
                class_conf_sum[cls] = class_conf_sum.get(cls, 0.0) + float(det["confidence"])
                class_conf_n[cls] = class_conf_n.get(cls, 0) + 1

        avg_confidence_per_class = {cls: (class_conf_sum[cls] / class_conf_n[cls]) for cls in class_counts}
        
        return {
            "total_frames": len(inference_results),
            "total_detections": total_detections,
            "frames_with_detections": frames_with_detections,
            "frames_without_detections": len(inference_results) - frames_with_detections,
            "detection_rate": frames_with_detections / len(inference_results) if inference_results else 0,
            "avg_detections_per_frame": total_detections / len(inference_results) if inference_results else 0,
            "class_counts": class_counts,
            "avg_confidence_per_class": avg_confidence_per_class,
            "unique_classes_detected": len(class_counts)
        }


def main():
    parser = argparse.ArgumentParser(description="Run inference pipeline on video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--video_id", help="Video ID (defaults to filename)")
    parser.add_argument("--model_key", type=str, default="models/raw-yolov8s-20251009-171047/best.pt", help="S3 key to model.pt")
    parser.add_argument("--bucket", default="visight-data-yusufmoola", help="S3 bucket name")
    parser.add_argument("--fps", type=int, help="Target FPS for frame extraction (default: all frames)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for batched inference (0 or 1 = per-frame)")
    parser.add_argument("--no_upload", action="store_true", help="Skip uploading to S3")
    parser.add_argument("--no_annotate", action="store_true", help="Skip saving annotated frames")
    
    args = parser.parse_args()    
    
    # Use filename as video_id if not provided
    if not args.video_id:
        args.video_id = Path(args.video).stem
    
    pipeline = InferencePipeline(
        s3_bucket=args.bucket, 
        model_key=args.model_key,
        fps=args.fps,
        confidence_threshold=args.confidence,
        batch_size=(args.batch_size if args.batch_size and args.batch_size > 1 else None)
    )
    
    result = pipeline.run_inference_on_video(
        args.video, 
        args.video_id, 
        upload_to_s3=not args.no_upload,
        save_annotated_frames=not args.no_annotate
    )
    
    print(f"\nPipeline complete. Processed {result['total_frames']} frames.")


if __name__ == "__main__":
    main()
