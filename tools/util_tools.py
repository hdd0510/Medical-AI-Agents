import cv2
import numpy as np
from PIL import Image
import os
import base64
import io
from typing import Dict, Any, List, Union
from tools.base_tools import Tool
class VisualizationTool(Tool):
    """Tool for visualizing medical images with detections, annotations, etc."""
    
    def __init__(self):
        super().__init__(
            name="visualize_image",
            description="Visualize medical images with annotations like bounding boxes, labels, and segmentations."
        )
    
    def run(self, 
           image_path: str,
           detections: List[Dict[str, Any]] = None,
           class_name: str = None,
           annotations: Dict[str, Any] = None,
           output_format: str = "base64") -> Dict[str, Any]:
        """
        Visualize an image with annotations.
        
        Args:
            image_path: Path to the image
            detections: List of detection objects with bbox, class_name, confidence
            class_name: Optional class name to display (e.g., from classifier)
            annotations: Additional annotations to display
            output_format: Format for output ('base64', 'save', 'array')
            
        Returns:
            Dict with visualization results
        """
        try:
            # Load image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return {"error": f"Image path not found: {image_path}", "success": False}
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path, np.ndarray):
                image = image_path.copy()
            elif isinstance(image_path, Image.Image):
                image = np.array(image_path)
            else:
                return {"error": "Unsupported image format", "success": False}
            
            # Make a copy to avoid modifying the original
            vis_image = image.copy()
            
            # Add detections if provided
            if detections:
                vis_image = self._draw_detections(vis_image, detections)
            
            # Add class name if provided
            if class_name:
                vis_image = self._add_text(vis_image, f"Class: {class_name}", position=(10, 30))
            
            # Add custom annotations if provided
            if annotations:
                if "text" in annotations:
                    for text_item in annotations["text"]:
                        vis_image = self._add_text(
                            vis_image, 
                            text_item["text"], 
                            position=text_item.get("position", (10, 60)),
                            color=text_item.get("color", (0, 255, 0)),
                            thickness=text_item.get("thickness", 2),
                            font_scale=text_item.get("font_scale", 0.8)
                        )
                
                if "lines" in annotations:
                    for line in annotations["lines"]:
                        pt1 = line["pt1"]
                        pt2 = line["pt2"]
                        color = line.get("color", (0, 255, 0))
                        thickness = line.get("thickness", 2)
                        cv2.line(vis_image, pt1, pt2, color, thickness)
                
                if "circles" in annotations:
                    for circle in annotations["circles"]:
                        center = circle["center"]
                        radius = circle["radius"]
                        color = circle.get("color", (0, 255, 0))
                        thickness = circle.get("thickness", 2)
                        cv2.circle(vis_image, center, radius, color, thickness)
            
            # Return based on output format
            if output_format == "base64":
                # Convert to PIL Image and then to base64
                pil_img = Image.fromarray(vis_image)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    "image_base64": img_str,
                    "width": vis_image.shape[1],
                    "height": vis_image.shape[0],
                    "success": True
                }
                
            elif output_format == "save":
                # Save to a file
                output_path = os.path.splitext(image_path)[0] + "_visualized.png" if isinstance(image_path, str) else "visualized_image.png"
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                return {
                    "output_path": output_path,
                    "width": vis_image.shape[1],
                    "height": vis_image.shape[0],
                    "success": True
                }
                
            elif output_format == "array":
                return {
                    "image_array": vis_image.tolist(),
                    "width": vis_image.shape[1],
                    "height": vis_image.shape[0],
                    "success": True
                }
                
            else:
                return {"error": f"Unsupported output format: {output_format}", "success": False}
                
        except Exception as e:
            import traceback
            return {
                "error": f"Visualization failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "success": False
            }
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and labels for detections."""
        for det in detections:
            # Get bbox coordinates
            if "bbox" in det:
                x1, y1, x2, y2 = map(int, det["bbox"])
            else:
                continue
                
            # Get class and confidence
            class_name = det.get("class_name", "object")
            confidence = det.get("confidence", 0)
            
            # Draw bbox
            color = (0, 255, 0)  # Green by default
            if confidence < 0.5:
                color = (255, 0, 0)  # Red for low confidence
            elif confidence < 0.7:
                color = (255, 255, 0)  # Yellow for medium confidence
                
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            self._add_text(image, label, (x1, y1 - 10), color=color)
        
        return image
    
    def _add_text(self, image: np.ndarray, text: str, position: tuple, 
                 color: tuple = (0, 255, 0), thickness: int = 2, 
                 font_scale: float = 0.8) -> np.ndarray:
        """Add text to image with background for better visibility."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(
            image, 
            (position[0], position[1] - text_height - baseline), 
            (position[0] + text_width, position[1] + baseline), 
            (0, 0, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            image, 
            text, 
            position, 
            font, 
            font_scale, 
            color, 
            thickness
        )
        
        return image
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image or image data"
            },
            "detections": {
                "type": "array",
                "description": "List of detection objects to visualize",
                "default": None
            },
            "class_name": {
                "type": "string",
                "description": "Optional class name to display",
                "default": None
            },
            "annotations": {
                "type": "object",
                "description": "Additional annotations to display (text, lines, circles)",
                "default": None
            },
            "output_format": {
                "type": "string",
                "description": "Format for output ('base64', 'save', 'array')",
                "default": "base64"
            }
        }
    
    def _get_returns(self) -> Dict[str, Any]:
        """Get the return schema for this tool."""
        return {
            "success": {
                "type": "boolean",
                "description": "Whether the visualization was successful"
            },
            "image_base64": {
                "type": "string",
                "description": "Base64-encoded visualized image (if output_format='base64')"
            },
            "output_path": {
                "type": "string",
                "description": "Path to saved image (if output_format='save')"
            },
            "width": {
                "type": "integer", 
                "description": "Width of visualized image"
            },
            "height": {
                "type": "integer",
                "description": "Height of visualized image"
            }
        }