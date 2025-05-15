import logging
import torch
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MedClassifierAgent1")

@dataclass
class ClassifierConfig1:
    """Configuration for the Medical Classifier Agent 1."""
    model_path: str
    name: str = "MedicalClassifierAgent1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.5
    num_classes: int = 10
    batch_size: int = 32

class MedicalClassifierAgent1:
    """
    A medical image classification agent specialized for classifying medical images
    into predefined categories.
    """
    
    def __init__(self, config: ClassifierConfig1):
        """Initialize the Medical Classifier Agent 1."""
        self.config = config
        self.name = config.name
        self.device = torch.device(config.device)
        logger.info(f"Initializing Medical Classifier Agent 1 on device: {self.device}")
        
        # Agent initialization state
        self.initialized = False
        
        # Initialize metrics
        self.metrics = {
            "total_images": 0,
            "correct_predictions": 0,
            "avg_confidence": 0.0,
            "avg_inference_time": 0.0
        }
        
        # Define class names (should be set externally if needed)
        self.class_names = []

    def initialize(self) -> bool:
        """
        Initialize the agent by loading the model and resources.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load the classification model
            self._load_model()
            
            # Mark as initialized
            self.initialized = True
            logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {str(e)}")
            return False

    def _load_model(self):
        """Load the classification model."""
        try:
            logger.info(f"Loading classification model from {self.config.model_path}")
            # Load your preferred classification model here (e.g., ResNet, EfficientNet)
            self.model = YOLO(self.config.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load classification model: {str(e)}")

    def classify(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Classify a medical image into predefined categories.
        
        Args:
            image: Path to image or PIL Image object
            
        Returns:
            Dict containing classification results and metadata
        """
        logger.info("Processing image for classification")
        
        try:
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Perform classification
            predictions = self._run_classification(image)
            
            # Update metrics
            self._update_metrics(predictions)
            
            return {
                "predictions": predictions,
                "top_prediction": max(predictions, key=lambda x: x["confidence"]),
                "image_size": image.size,
                "metrics": self.metrics
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {
                "error": str(e),
                "predictions": [],
                "top_prediction": None
            }

    def _run_classification(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run classification on the image and return results."""
        # Implement your classification logic here
        # This is a placeholder implementation
        return []

    def _update_metrics(self, predictions: List[Dict[str, Any]]):
        """Update the agent's performance metrics."""
        self.metrics["total_images"] += 1
        
        if predictions:
            confidences = [p.get("confidence", 0) for p in predictions]
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

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image for classification.
        
        Args:
            input_data: Dictionary containing the input data with at least 'image_path' key
            
        Returns:
            Dict containing classification results and metadata
        """
        if not self.initialized:
            return {"success": False, "error": "Agent has not been initialized"}
        
        # Extract image path
        image_path = input_data.get("image_path")
        if not image_path:
            return {"success": False, "error": "No image_path provided in input_data"}
        
        try:
            # Classify the image
            result = self.classify(image_path)
            
            # Add standard fields
            result["success"] = "error" not in result
            
            # Add specific class info if available
            if result.get("top_prediction") and self.class_names:
                class_id = result["top_prediction"].get("class_id", 0)
                confidence = result["top_prediction"].get("confidence", 0.0)
                
                if 0 <= class_id < len(self.class_names):
                    result["class_id"] = class_id
                    result["class_name"] = self.class_names[class_id]
                    result["confidence"] = confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Process failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 