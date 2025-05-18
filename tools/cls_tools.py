import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union
import os
from tools.base_tools import Tool

class ImageClassifierTool(Tool):
    """Tool for image classification using various models."""
    
    def __init__(self, 
                model_path: str, 
                class_names: List[str],
                device: str = "cuda",
                input_size: tuple = (224, 224),
                model_type: str = "yolo"):
        
        super().__init__(
            name="image_classifier",
            description=f"Classify images into {len(class_names)} categories: {', '.join(class_names)}"
        )
        self.model_path = model_path
        self.class_names = class_names
        self.device = device
        self.input_size = input_size
        self.model_type = model_type
        self.model = None
        self.class_map = {i: name for i, name in enumerate(class_names)}
        self._load_model()
        
    def _load_model(self):
        """Load the classification model."""
        try:
            if self.model_type == "yolo":
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
            elif self.model_type == "torchvision":
                import torchvision.models as models
                # This is just an example - you'd need to customize based on your model
                self.model = models.efficientnet_b0(pretrained=False)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = torch.nn.Linear(num_ftrs, len(self.class_names))
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                # Define transforms
                import torchvision.transforms as transforms
                self.transform = transforms.Compose([
                    transforms.Resize(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {str(e)}")
    
    def run(self, 
           image_path: str, 
           confidence_threshold: float = 0.5,
           top_k: int = 3) -> Dict[str, Any]:
        """
        Run classification on an image.
        
        Args:
            image_path: Path to the image
            confidence_threshold: Classification confidence threshold
            top_k: Return top k predictions
            
        Returns:
            Dict with classification results
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
            
            # Classify based on model type
            if self.model_type == "yolo":
                predictions = self._classify_with_yolo(image, confidence_threshold)
            elif self.model_type == "torchvision":
                predictions = self._classify_with_torchvision(image, confidence_threshold)
            else:
                return {"error": f"Unsupported model type: {self.model_type}", "success": False}
            
            # Get top-k predictions
            top_predictions = predictions[:min(top_k, len(predictions))]
            
            # Get top-1 prediction
            top_prediction = top_predictions[0] if top_predictions else {"class": "unknown", "confidence": 0}
            
            return {
                "class": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "predictions": top_predictions,
                "image_path": image_path if isinstance(image_path, str) else "in-memory-image",
                "success": True
            }
                
        except Exception as e:
            import traceback
            return {
                "error": f"Classification failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "success": False
            }
    
    def _classify_with_yolo(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Classify using YOLO model."""
        results = self.model.predict(
            source=image,
            verbose=False
        )
        
        # Process results
        predictions = []
        for result in results:
            # Get probabilities
            probs = result.probs.data.cpu().numpy()
            
            # Sort indices by probability (descending)
            indices = np.argsort(probs)[::-1]
            
            for i, idx in enumerate(indices):
                conf = float(probs[idx])
                if conf >= confidence_threshold:
                    class_id = int(idx)
                    class_name = self.class_map.get(class_id, f"class_{class_id}")
                    
                    predictions.append({
                        "rank": i + 1,
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": conf
                    })
        
        return predictions
    
    def _classify_with_torchvision(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Classify using torchvision model."""
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Sort indices by probability (descending)
        indices = np.argsort(probs)[::-1]
        
        # Create predictions list
        predictions = []
        for i, idx in enumerate(indices):
            conf = float(probs[idx])
            if conf >= confidence_threshold:
                class_id = int(idx)
                class_name = self.class_map.get(class_id, f"class_{class_id}")
                
                predictions.append({
                    "rank": i + 1,
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": conf
                })
        
        return predictions
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to classify"
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence threshold for classification results",
                "default": 0.5
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top predictions to return",
                "default": 3
            }
        }
    
    def _get_returns(self) -> Dict[str, Any]:
        """Get the return schema for this tool."""
        return {
            "class": {
                "type": "string",
                "description": "Top predicted class name"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score for the top prediction"
            },
            "predictions": {
                "type": "array",
                "description": "List of top-k predictions with class and confidence"
            },
            "success": {
                "type": "boolean",
                "description": "Whether the classification was successful"
            }
        }