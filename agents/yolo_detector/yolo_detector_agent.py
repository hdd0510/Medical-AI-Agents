import logging
import torch
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLODetectorAgent")

@dataclass
class YOLOConfig:
    """Configuration for the YOLO Detector Agent."""
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: List[int] = None  # None means all classes
    img_size: int = 640

class YOLODetectorAgent:
    """
    A medical object detection agent using YOLO model.
    Specialized for detecting medical conditions and anatomical structures.
    """
    
    def __init__(self, config: YOLOConfig):
        """Initialize the YOLO Detector Agent."""
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Initializing YOLO Detector Agent on device: {self.device}")
        
        # Load YOLO model
        self._load_model()
        
        # Initialize metrics
        self.metrics = {
            "total_images": 0,
            "total_detections": 0,
            "avg_confidence": 0.0,
            "avg_inference_time": 0.0
        }

    def _load_model(self):
        """Load the YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def detect(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Detect medical conditions and anatomical structures in an image.
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            Dict containing detection results and metadata
        """
        logger.info("Processing image for detection")
        
        try:
            # Convert image to numpy array if needed
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image = np.array(image)
            
            # Perform detection
            results = self.model(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.classes,
                max_det=self.config.max_detections,
                imgsz=self.config.img_size
            )
            
            # Process results
            detections = self._process_results(results[0])
            
            # Update metrics
            self._update_metrics(detections)
            
            return {
                "detections": detections,
                "num_detections": len(detections),
                "image_size": image.shape[:2],
                "metrics": self.metrics
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return {
                "error": str(e),
                "detections": [],
                "num_detections": 0
            }

    def _process_results(self, result) -> List[Dict[str, Any]]:
        """Process YOLO detection results into a standardized format."""
        detections = []
        
        for box in result.boxes:
            detection = {
                "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])]
            }
            detections.append(detection)
        
        return detections

    def _update_metrics(self, detections: List[Dict[str, Any]]):
        """Update the agent's performance metrics."""
        self.metrics["total_images"] += 1
        self.metrics["total_detections"] += len(detections)
        
        if detections:
            confidences = [d["confidence"] for d in detections]
            self.metrics["avg_confidence"] = sum(confidences) / len(confidences)

    def get_metrics(self) -> Dict[str, Any]:
        """Get the agent's performance metrics."""
        return self.metrics

    def export_metrics(self, output_path: str):
        """Export the agent's metrics to a file."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics exported to {output_path}")

    def visualize_detections(self, image: Union[str, Image.Image, np.ndarray], 
                           detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with visualized detections
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            class_name = det["class_name"]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image 