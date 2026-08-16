import torch
from ultralytics import YOLO
import numpy as np

from config import MODEL_SIZE, CONFIDENCE_THRESHOLD

class ObjectDetector:
    def __init__(self, model_size: str = MODEL_SIZE, confidence: float = CONFIDENCE_THRESHOLD):
        """
        Initialize YOLOv8 object detector
        model_size: yolov8n.pt (nano/fastest) or yolov8s.pt (small/better accuracy)
        confidence: minimum confidence threshold
        """
        self.model = YOLO(model_size)
        self.confidence = confidence

        # ADAS relevant classes from COCO dataset
        self.relevant_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic light",
            11: "stop sign"
        }

        print(f"Object detector loaded: {model_size}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run object detection on a frame
        Returns list of detections: [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        results = self.model(frame, verbose=False, conf=self.confidence)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])

                # Only keep ADAS relevant objects
                if class_id not in self.relevant_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_name = self.relevant_classes[class_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })

        return detections
