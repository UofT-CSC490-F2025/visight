import argparse
from pathlib import Path
import json
import tempfile
import cv2
from pipelines.inference.video_processor import VideoProcessor
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path
from pipelines.clients.s3_client import s3Client
from typing import Optional
import logging
import math
import cProfile
import pstats
from io import StringIO
from datetime import datetime, timezone

class InferencePipeline:
    """
    Main inference pipeline for video logo detection
    """
    
    def __init__(self, model_path: str, fps: int = None, 
                 confidence_threshold: float = 0.5, logger:Optional[logging.Logger] = None, batch_size: int = 80):
        self.video_processor = VideoProcessor(fps=fps)
        self.confidence_threshold = confidence_threshold        
        self.model = YOLO(model_path)
        self.logger = logger
        self.batch_size = batch_size
    
    def run_inference_on_video(self, video_path: str, video_id: str, 
                               s3_bucket: Optional[str] = None, 
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
        msg = f"\n{'='*60}\nProcessing video: {video_id}\n{'='*60}\n"
        if self.logger: self.logger.info(msg)
        else: print(msg)
        
        # Step 1: Get video info
        if self.logger: self.logger.info("Step 1: Extracting video metadata...")
        else: print("Step 1: Extracting video metadata...")
        video_info = self.video_processor.get_video_info(video_path)
        if self.logger: self.logger.info(f"Video info: {video_info}")
        else: print(f"Video info: {video_info}")
        
        # Step 2: Upload original video to S3
        s3_client = s3Client(buckets=[s3_bucket]) if s3_bucket else None
        if s3_client:
            if self.logger: self.logger.info("\nStep 2: Uploading video to S3...")
            else: print("\nStep 2: Uploading video to S3...")
            video_key = f"inference/videos/{video_id}"
            video_s3_path = s3_client.upload_file(
                video_path, 
                video_key, 
                content_type='video/mp4'
            )
            if self.logger: self.logger.info(f"Video uploaded to: {video_s3_path}")
            else: print(f"Video uploaded to: {video_s3_path}")
        else:
            video_s3_path = None
        
        # Step 3: Extract frames
        if self.logger: self.logger.info("\nStep 3: Extracting frames...")
        else: print("\nStep 3: Extracting frames...")
        data_dir = "/data"
        frames_dir = Path(data_dir) / "frames"
        annotated_dir = Path(data_dir) / "annotated" if save_annotated_frames else None
        if annotated_dir:
            annotated_dir.mkdir(parents=True, exist_ok=True)
        
        frames_metadata = self.video_processor.extract_frames(video_path, str(frames_dir))
        
        # Step 4: Upload original frames to S3
        if s3_client:
            if self.logger: self.logger.info("\nStep 4: Uploading original frames to S3...")
            else: print("\nStep 4: Uploading original frames to S3...")
            s3_frames = []
            for frame_meta in tqdm(frames_metadata, desc="Uploading frames"):
                frame_path = Path(frame_meta["file_path"])
                frame_key = f"inference/frames/{video_id}/original/{frame_path.name}"
                
                frame_s3_path = s3_client.upload_file(
                    frame_path,
                    frame_key,
                    content_type='image/jpeg'
                )
                
                s3_frames.append({
                    **frame_meta,
                    "s3_path": frame_s3_path
                })
            if self.logger: self.logger.info(f"Uploaded {len(s3_frames)} frames to S3")
            else: print(f"Uploaded {len(s3_frames)} frames to S3")
        else:
            s3_frames = frames_metadata
        
        # Step 5: Run inference on frames
        if self.logger: self.logger.info("\nStep 5: Running inference on frames...")
        else: print("\nStep 5: Running inference on frames...")
        inference_results = self._run_model_inference(
            frames_metadata, 
            frames_dir, 
            annotated_dir
        )
        print("INFERENCE RESULTS:", len(inference_results))
        
        # Step 6: Upload annotated frames to S3
        if s3_client and save_annotated_frames and annotated_dir:
            if self.logger: self.logger.info("\nStep 6: Uploading annotated frames to S3...")
            else: print("\nStep 6: Uploading annotated frames to S3...")
            for result in tqdm(inference_results, desc="Uploading annotated frames"):
                if result.get("annotated_frame_path"):
                    annotated_path = Path(result["annotated_frame_path"])
                    annotated_key = f"inference/frames/{video_id}/annotated/{annotated_path.name}"
                    
                    annotated_s3_path = s3_client.upload_file(
                        annotated_path,
                        annotated_key,
                        content_type='image/jpeg'
                    )
                    result["annotated_s3_path"] = annotated_s3_path
            
            if self.logger: self.logger.info(f"Uploaded {len(inference_results)} annotated frames to S3")
            else: print(f"Uploaded {len(inference_results)} annotated frames to S3")
        
        # Step 7: Generate summary statistics
        print("\nStep 7: Generating summary statistics...")
        summary_stats = self._generate_summary_stats(inference_results)
        
        # Step 8: Aggregate results
        if self.logger: self.logger.info("\nStep 8: Aggregating results...")
        else: print("\nStep 8: Aggregating results...")
        pipeline_results = {
            "video_id": video_id,
            "video_info": video_info,
            "video_s3_path": video_s3_path,
            "confidence_threshold": self.confidence_threshold,
            "total_frames": len(frames_metadata),
            "frames": s3_frames if s3_bucket else frames_metadata,
            "inference_results": inference_results,
            "summary_stats": summary_stats
        }
        res_summary_path = "results.json"
        with open(res_summary_path, 'w') as json_file:
            filter_results = pipeline_results
            new_frames = []
            for f in pipeline_results["frames"]:
                f1 = f.copy()
                f1.pop("annotated_frame_path", None)
                new_frames.append(f1)
            filter_results["frames"] = new_frames
            json.dump(filter_results, json_file, indent=4)

        
        # Save results
        if s3_client:
                results_key = f"inference/results/{video_id}/results.json"
                results_bytes = json.dumps(pipeline_results, indent=2).encode('utf-8')
                results_s3_path = s3_client.put_object(
                    results_key,
                    results_bytes,
                    content_type='application/json'
                )
                if self.logger: self.logger.info(f"\nResults saved to: {results_s3_path}")
                else: print(f"\nResults saved to: {results_s3_path}")
                pipeline_results["results_s3_path"] = results_s3_path
        
        msg = f"\n{'='*60}\nPipeline complete for video: {video_id}\nTotal detections: {summary_stats['total_detections']}"
        if self.logger: self.logger.info(msg)
        else: print(msg)
        msg = f"Frames with detections: {summary_stats['frames_with_detections']}, out of {summary_stats['total_frames']} frames\n{'='*60}\n"
        if self.logger: self.logger.info(msg)
        else: print(msg)
        
        res_dirs = ([annotated_dir] if annotated_dir else []) + [frames_dir.parent / "profiling"] # [Path("/data/profiling")]
        return res_dirs, res_summary_path
    
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
        """
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            batches = self._split_into_batches(frames_metadata)
            results = []
            for i in range(len(batches)): 
                results.extend(self._minibatch_inference(i, batches[i]["paths"], batches[i]["frame_meta"], annotated_dir))
            
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
            # with open(profile_file, 'w') as f:
            #     f.write(s.getvalue())
            profile_file.write_text(s.getvalue())
            print(f"\n[PROFILING] _run_model_inference profile saved to: {profile_file}")
    
    def _split_into_batches(self, frames_metadata: list): 
        batches = []
        num_batches = math.ceil(len(frames_metadata)/self.batch_size)
        for i in range(num_batches): 
            start_i = i*self.batch_size
            end_i = min((i + 1) * self.batch_size, len(frames_metadata))
            metadata_slice = frames_metadata[start_i:end_i]
            batches.append({"paths":[f["file_path"] for f in metadata_slice],
                            "frame_meta": metadata_slice})
        
        return batches
    
    def _minibatch_inference(self, batch_index: int, frame_paths: list[str], file_meta: list[dict], annotated_dir: Path = None):
        # profiler = cProfile.Profile()
        # profiler.enable()
        # try:
        all_dets = self.model.predict(frame_paths, conf=self.confidence_threshold, verbose=False, stream=True)
        # finally:
        #     profiler.disable()
            
        #     # Save profiling stats
        #     s = StringIO()
        #     ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        #     ps.print_stats(50)  # Top 50 functions
            
        #     # Save profile to frames directory parent
        #     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        #     profile_output_dir = Path("/data/profiling")
        #     profile_output_dir.mkdir(parents=True, exist_ok=True)
        #     profile_file = profile_output_dir / f"minibatch_inference_{timestamp}.txt"
        #     profile_file.write_text(s.getvalue())
        #     print(f"\n[PROFILING] minibatch_inference profile saved to: {profile_file}")
        return self._process_inference_res(batch_index, all_dets, file_meta, annotated_dir)
    
    def _process_inference_res(self, batch_index: int, all_dets, file_meta: list[dict], annotated_dir: Path = None): 
        # profiler = cProfile.Profile()
        # profiler.enable()
        # try:
        results = []
        i = 0
        for det in tqdm(all_dets, total=len(file_meta), desc=f"Batch {batch_index} Processing"): 
            name_map = det.names
            detection_data = det.boxes.data.tolist()
            detection_info = [{"bbox": d[:4], "confidence":d[4], "class_name":name_map[d[5]]} for d in detection_data]
            result = {
                "frame_id": file_meta[i]["frame_id"],
                "frame_number": file_meta[i]["frame_number"],
                "timestamp": file_meta[i]["timestamp"],
                "detections": detection_info,
                "detection_count": len(detection_info)
            }
            if annotated_dir and len(detection_info) > 0:
                annotated_path = self._draw_bounding_boxes(
                    Path(file_meta[i]["file_path"]), 
                    detection_info, 
                    annotated_dir
                )
                result["annotated_frame_path"] = str(annotated_path)
                
            results.append(result)
            i += 1
            
        return results
        # finally:
        #     profiler.disable()
            
        #     # Save profiling stats
        #     s = StringIO()
        #     ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        #     ps.print_stats(50)  # Top 50 functions
            
        #     # Save profile to frames directory parent
        #     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        #     profile_output_dir = Path("/data/profiling")
        #     profile_output_dir.mkdir(parents=True, exist_ok=True)
        #     profile_file = profile_output_dir / f"inference_res_processing_{timestamp}.txt"
        #     profile_file.write_text(s.getvalue())
        #     print(f"\n[PROFILING] _run_model_inference profile saved to: {profile_file}")
    
    @profiled(name="draw_bounding_boxes", stats_limit=50)
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
    
    @profiled(name="generate_summary_stats", stats_limit=50)
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
