from typing import Any
import logging
import os
import json
from collections import defaultdict
import uuid

class Detector:
    def _detectron2_predictor(self):
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        model_name = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"

        cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)

        return cfg, predictor


    def detect_objects(self, video_path: str, output_dir: str = "output") -> list[dict[str, Any]]:
        """
        Detect objects in a video and save individual frames with detected objects.

        Args:
            video_path (str): The path to the video file.
            output_dir (str, optional): Directory to save output frames. Defaults to "output".

        Returns:
            Dict[str, Any]: A dictionary of detected objects.
        """
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        import cv2
        
        logging.info(f"Starting object detection on video: {video_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        cfg, predictor = self._detectron2_predictor()
        logging.info("Initialized Detectron2 predictor")
        
        # Get metadata for class names
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        
        video_reader = cv2.VideoCapture(video_path)
        if not video_reader.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")
        logging.info("Opened video file successfully")
        
        # Initialize dictionary to store timestamps
        object_timestamps = defaultdict(list)
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video FPS: {fps}")
        
        frame_count = 0
        processed_frames = 0
        frame_interval = 5  # Process every 5th frame
        file_prefix = str(uuid.uuid4())
        
        while True:
            ret, frame = video_reader.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames that aren't at the interval
            if frame_count % frame_interval != 0:
                continue
                
            timestamp = frame_count / fps  # Convert frame number to seconds
            logging.debug(f"Processing frame {frame_count} at {timestamp:.2f}s")
            
            predictions = predictor(frame)
            instances = predictions["instances"]
            
            # Only process frames with detected objects
            if len(instances) > 0:
                processed_frames += 1
                logging.info(f"Frame {frame_count}: Found {len(instances)} objects")
                
                # Visualize predictions on frame
                v = Visualizer(frame[:,:,::-1], metadata, scale=1.2)
                out = v.draw_instance_predictions(instances.to("cpu"))
                annotated_frame = out.get_image()[:,:,::-1]
                
                # Save frame for each detected object
                for idx in range(len(instances)):
                    if instances.has("pred_classes"):
                        # Get class name from predicted class ID
                        class_id = instances.pred_classes[idx].item()
                        obj_name = metadata.thing_classes[class_id]
                        
                        output_filename = f"out-{file_prefix}-{frame_count}-{obj_name}-{idx+1}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        cv2.imwrite(output_path, annotated_frame)
                        logging.debug(f"Saved annotated frame to: {output_path}")
                        
                        # Store timestamp and filename for this object
                        object_timestamps[obj_name].append({
                            "frame": frame_count,
                            "timestamp": round(timestamp, 2),
                            "filename": output_filename
                        })
        
        video_reader.release()
        
        # Save timestamps to JSON file with file ID prefix
        json_filename = f"object_timestamps-{file_prefix}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(object_timestamps, f, indent=2)
        logging.info(f"Saved object timestamps to {json_path}")
        
        logging.info(f"Completed processing. Processed {processed_frames} frames with detections out of {frame_count} total frames")
        return object_timestamps
