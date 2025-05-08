#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical Image Classifier Agent - 10 Classes
------------------------------------------
Agent chuyên biệt cho việc phân loại hình ảnh nội soi tiêu hóa thành 10 lớp
"""

import os
import sys
import json
import logging
import time
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional, Tuple

from agents.base_agent import BaseAgent, BaseAgentConfig

@dataclass
class RegionClassifierClassConfig(BaseAgentConfig):
    """Cấu hình cho Classifier 10 lớp."""
    # Tham số model
    input_size: Tuple[int, int] = (224, 224)  # Kích thước input cho model phân loại
    confidence_threshold: float = 0.5        # Ngưỡng độ tin cậy
    
    # Thông tin classes
    classes: List[str] = field(default_factory=lambda: [
        "1_Hau_hong", "2_Thuc_quan", "3_Tam_vi", "4_Than_vi", 
        "5_Phinh_vi", "6_Hang_vi", "7_Bo_cong_lon", "8_Bo_cong_nho", 
        "9_Hanh_ta_trang", "10_Ta_trang"
    ])
    
    # Cache
    enable_cache: bool = True                # Bật cache cho kết quả
    cache_expiry: int = 3600                 # Thời gian hết hạn cache (giây)
    
    # Augmentation khi inference
    tta_enabled: bool = False                # Test-time augmentation
    
    # Threshold đánh dấu không chắc chắn
    uncertain_threshold: float = 0.3         # Ngưỡng để đánh dấu "không chắc chắn"


class RegionClassifierClassAgent(BaseAgent):
    """
    Agent phân loại hình ảnh nội soi tiêu hóa thành 10 lớp.
    
    Phân loại vị trí trong đường tiêu hóa từ hình ảnh nội soi.
    """
    
    def __init__(self, config: RegionClassifierClassConfig):
        """
        Khởi tạo Classifier Agent.
        
        Args:
            config: Cấu hình cho agent
        """
        super().__init__(config)
        self.classifier_config = config
        self.device = torch.device(config.device)
        
        # Model phân loại
        self.model = None
        
        # Cache kết quả
        self.result_cache = {}
        
        # Tên lớp và ánh xạ
        self.class_names = self.classifier_config.classes
        self.class_map = {i: name for i, name in enumerate(self.class_names)}
        
        # Vị trí giải phẫu tương ứng với từng lớp (thông tin thêm)
        self.class_descriptions = {
            "1_Hau_hong": "Họng sau - Phần sau của cổ họng",
            "2_Thuc_quan": "Thực quản - Ống nối từ họng đến dạ dày",
            "3_Tam_vi": "Tâm vị - Phần trên của dạ dày, nối với thực quản",
            "4_Than_vi": "Thân vị - Phần thân dạ dày",
            "5_Phinh_vi": "Phình vị - Phần phình của dạ dày",
            "6_Hang_vi": "Hang vị - Phần dưới của dạ dày",
            "7_Bo_cong_lon": "Bờ cong lớn - Bờ bên trái của dạ dày",
            "8_Bo_cong_nho": "Bờ cong nhỏ - Bờ bên phải của dạ dày",
            "9_Hanh_ta_trang": "Hành tá tràng - Phần đầu của tá tràng",
            "10_Ta_trang": "Tá tràng - Phần đầu của ruột non"
        }
        
        # Metrics bổ sung
        self.metrics.update({
            "class_distribution": {cls: 0 for cls in self.class_names},
            "cache_hits": 0,
            "cache_misses": 0,
            "uncertain_classifications": 0,
            "avg_confidence": 0,
            "total_confidence": 0
        })
        
        self.logger.info(f"Classifier 10 lớp đã được khởi tạo với cấu hình: {self.classifier_config}")
    
    def initialize(self) -> bool:
        """
        Khởi tạo Classifier Agent, tải model phân loại.
        
        Returns:
            bool: True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            self.logger.info(f"Đang tải model phân loại từ {self.classifier_config.model_path}")
            
            # Kiểm tra đường dẫn model
            if not os.path.exists(self.classifier_config.model_path) and not self.classifier_config.model_path.startswith(("http://", "https://")):
                self.logger.warning(f"Không tìm thấy model tại {self.classifier_config.model_path}. Kiểm tra lại đường dẫn.")
            
            # Phương pháp 1: Sử dụng Ultralytics YOLOv8-cls
            try:
                from ultralytics import YOLO
                
                # Tải model
                self.model = YOLO(self.classifier_config.model_path)
                
                # Chuyển model sang device thích hợp
                self.model.to(self.device)
                
                self.logger.info(f"Model phân loại đã được tải thành công trên thiết bị {self.device}")
                self.initialized = True
                return True
                
            except ImportError:
                self.logger.warning("Không thể sử dụng Ultralytics. Thử phương pháp PyTorch")
                
                # Phương pháp 2: Sử dụng PyTorch trực tiếp
                try:
                    import torch
                    import torchvision.models as models
                    import torchvision.transforms as transforms
                    
                    # Tải pretrained model và chỉnh sửa lớp cuối
                    self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
                    num_ftrs = self.model.classifier[1].in_features
                    self.model.classifier[1] = torch.nn.Linear(num_ftrs, len(self.class_names))
                    
                    # Tải weights
                    self.model.load_state_dict(torch.load(self.classifier_config.model_path, map_location=self.device))
                    self.model.to(self.device)
                    self.model.eval()
                    
                    # Định nghĩa transform
                    self.transform = transforms.Compose([
                        transforms.Resize(self.classifier_config.input_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    self.logger.info(f"Model PyTorch đã được tải thành công")
                    self.initialized = True
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Không thể tải model PyTorch: {str(e)}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Classifier Agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý yêu cầu phân loại hình ảnh.
        
        Args:
            input_data: Dictionary chứa:
                - image_path: Đường dẫn đến hình ảnh hoặc URL
                - confidence (tùy chọn): Ghi đè ngưỡng độ tin cậy
                - detections (tùy chọn): Kết quả phát hiện từ detector agent
                
        Returns:
            Dictionary chứa kết quả và metadata
        """
        if not self.initialized:
            return {"error": "Classifier Agent chưa được khởi tạo", "success": False}
        
        # Trích xuất dữ liệu đầu vào
        image_path = input_data.get("image_path")
        
        # Kiểm tra đầu vào
        if not image_path:
            return {"error": "Thiếu đường dẫn hình ảnh (image_path)", "success": False}
        
        # Tạo cache key nếu caching được bật
        cache_key = None
        if self.classifier_config.enable_cache:
            confidence = input_data.get("confidence", self.classifier_config.confidence_threshold)
            
            cache_key = f"{image_path}:{confidence}"
            cached_result = self._check_cache(cache_key)
            
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
            
            self.metrics["cache_misses"] += 1
        
        # Thực hiện quá trình xử lý với monitoring
        result = self.run_with_monitoring(self._process_classification, image_path, input_data)
        
        # Lưu vào cache nếu cần
        if self.classifier_config.enable_cache and cache_key and "error" not in result:
            self._add_to_cache(cache_key, result)
        
        # Lưu vào bộ nhớ ngắn hạn
        self.remember("last_classification_result", result)
        self.remember(f"classification_result_{image_path}", result)
        
        return result
    
    def _process_classification(self, image_path: Union[str, Image.Image], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý nội bộ cho yêu cầu phân loại hình ảnh.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            input_data: Tham số bổ sung
            
        Returns:
            Dictionary chứa kết quả phân loại và metadata
        """
        try:
            # Lấy tham số từ input_data hoặc sử dụng giá trị mặc định
            confidence_threshold = input_data.get("confidence", self.classifier_config.confidence_threshold)
            
            # Tải hình ảnh
            image = self._load_image(image_path)
            
            # Thực hiện phân loại dựa trên loại model
            if hasattr(self.model, 'predict'):
                # Nếu sử dụng Ultralytics
                predictions = self._classify_with_ultralytics(image, confidence_threshold)
            else:
                # Nếu sử dụng PyTorch
                predictions = self._classify_with_pytorch(image, confidence_threshold)
            
            # Xử lý kết quả
            top_prediction = predictions[0] if predictions else {"class": "unknown", "confidence": 0}
            
            # Kiểm tra độ tin cậy
            is_uncertain = top_prediction["confidence"] < self.classifier_config.uncertain_threshold
            if is_uncertain:
                self.metrics["uncertain_classifications"] += 1
            
            # Cập nhật metrics
            class_name = top_prediction["class"]
            if class_name in self.metrics["class_distribution"]:
                self.metrics["class_distribution"][class_name] += 1
            
            # Tính toán độ tin cậy trung bình
            conf = top_prediction["confidence"]
            self.metrics["total_confidence"] += conf
            self.metrics["avg_confidence"] = (
                self.metrics["total_confidence"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            )
            
            # Thêm thông tin giải phẫu nếu có
            if class_name in self.class_descriptions:
                top_prediction["description"] = self.class_descriptions[class_name]
            
            return {
                "class": class_name,
                "confidence": top_prediction["confidence"],
                "predictions": predictions,
                "uncertain": is_uncertain,
                "image_path": image_path if isinstance(image_path, str) else "in-memory-image",
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý phân loại: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": f"Không thể xử lý yêu cầu: {str(e)}",
                "success": False
            }
    
    def _load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """
        Tải hình ảnh từ đường dẫn hoặc URL.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            
        Returns:
            PIL.Image: Hình ảnh đã tải
        """
        # Xử lý các loại đầu vào khác nhau
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                # Tải hình ảnh từ URL
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Tải hình ảnh từ đường dẫn cục bộ
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Không tìm thấy file hình ảnh: {image_path}")
                image = Image.open(image_path).convert('RGB')
        else:
            # Đã là PIL Image
            image = image_path.convert('RGB')
        
        return image
    
    def _classify_with_ultralytics(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Phân loại hình ảnh sử dụng Ultralytics YOLOv8-cls.
        
        Args:
            image: PIL Image
            confidence_threshold: Ngưỡng độ tin cậy
            
        Returns:
            List[Dict]: Danh sách các lớp dự đoán với độ tin cậy
        """
        # Thực hiện dự đoán
        results = self.model.predict(
            source=image,
            verbose=False
        )
        
        # Xử lý kết quả
        predictions = []
        
        for result in results:
            # Lấy probs và class
            probs = result.probs.data.cpu().numpy()
            
            # Sắp xếp theo độ tin cậy giảm dần
            indices = np.argsort(probs)[::-1]
            
            for i, idx in enumerate(indices):
                conf = float(probs[idx])
                
                # Chỉ thêm vào kết quả nếu vượt ngưỡng độ tin cậy
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
    
    def _classify_with_pytorch(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Phân loại hình ảnh sử dụng PyTorch.
        
        Args:
            image: PIL Image
            confidence_threshold: Ngưỡng độ tin cậy
            
        Returns:
            List[Dict]: Danh sách các lớp dự đoán với độ tin cậy
        """
        # Resize và transform
        if hasattr(self, 'transform'):
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # Fallback transform
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(self.classifier_config.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Thực hiện dự đoán
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Sắp xếp theo độ tin cậy giảm dần
        indices = np.argsort(probs)[::-1]
        
        # Xử lý kết quả
        predictions = []
        
        for i, idx in enumerate(indices):
            conf = float(probs[idx])
            
            # Chỉ thêm vào kết quả nếu vượt ngưỡng độ tin cậy
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
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Kiểm tra cache cho kết quả đã tính toán trước đó."""
        if not self.classifier_config.enable_cache:
            return None
            
        cached_item = self.result_cache.get(key)
        if not cached_item:
            return None
            
        # Kiểm tra thời gian hết hạn
        if time.time() - cached_item["timestamp"] > self.classifier_config.cache_expiry:
            # Cache đã hết hạn
            del self.result_cache[key]
            return None
            
        self.logger.info(f"Đã tìm thấy kết quả trong cache cho key: {key}")
        return cached_item["result"]
    
    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """Thêm kết quả vào cache."""
        if not self.classifier_config.enable_cache:
            return
            
        self.result_cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Log cache stats
        self.logger.debug(f"Cache hiện tại có {len(self.result_cache)} mục")
    
    def clear_cache(self):
        """Xóa tất cả các mục trong cache."""
        cache_size = len(self.result_cache)
        self.result_cache.clear()
        self.logger.info(f"Đã xóa {cache_size} mục trong cache")
    
    def batch_process(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Xử lý hàng loạt các yêu cầu phân loại hình ảnh.
        
        Args:
            batch_data: Danh sách các yêu cầu
            
        Returns:
            List[Dict]: Danh sách kết quả
        """
        results = []
        
        self.logger.info(f"Đang xử lý hàng loạt {len(batch_data)} yêu cầu")
        
        for request in batch_data:
            # Xử lý từng yêu cầu
            result = self.process(request)
            results.append(result)
        
        return results