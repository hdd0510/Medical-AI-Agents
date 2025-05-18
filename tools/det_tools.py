import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union
import os
from tools.base_tools import Tool

class YOLOTool(Tool):
    """Tool for object detection using YOLO."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(
            name="yolo_detect",
            description="Detect objects in an image using YOLO model. Specialized for medical polyp detection."
        )
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def run(self, 
           image_path: str, 
           confidence_threshold: float = 0.25, 
           iou_threshold: float = 0.45,
           max_detections: int = 100) -> Dict[str, Any]:
        """
        Run detection on an image.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to return
            
        Returns:
            Dict with detection results
        """
        if not self.model:
            return {"error": "Model not loaded", "success": False}
            
        try:
            # Load image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return {"error": f"Image path not found: {image_path}", "success": False}
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
                
            # Run detection
            results = self.model.predict(
                source=image,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = result.names[cls_id]
                    
                    # Calculate additional metrics
                    x1, y1, x2, y2 = xyxy.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    area = width * height
                    
                    detection = {
                        "bbox": xyxy.tolist(),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "width": width,
                        "height": height,
                        "center": [center_x, center_y],
                        "area": area
                    }
                    detections.append(detection)
            
            # Sort by confidence
            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            
            return {
                "detections": detections,
                "count": len(detections),
                "image_path": image_path if isinstance(image_path, str) else "in-memory-image",
                "success": True
            }
                
        except Exception as e:
            import traceback
            return {
                "error": f"Detection failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "success": False
            }
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to detect objects in"
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence threshold for detections (0.0-1.0)",
                "default": 0.25
            },
            "iou_threshold": {
                "type": "number",
                "description": "IoU threshold for Non-Maximum Suppression",
                "default": 0.45
            },
            "max_detections": {
                "type": "integer",
                "description": "Maximum number of detections to return",
                "default": 100
            }
        }
    
    def _get_returns(self) -> Dict[str, Any]:
        """Get the return schema for this tool."""
        return {
            "detections": {
                "type": "array",
                "description": "List of detected objects with their properties"
            },
            "count": {
                "type": "integer",
                "description": "Number of detected objects"
            },
            "success": {
                "type": "boolean",
                "description": "Whether the detection was successful"
            }
        }