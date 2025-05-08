#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical Object Detector Agent (Single Class)
-------------------------------------------
Agent chuyên biệt cho việc phát hiện đối tượng trong hình ảnh y tế sử dụng YOLOv8 với một class duy nhất.
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
class DetectorAgentConfig(BaseAgentConfig):
    """Cấu hình cho Detector Agent với YOLOv8."""
    # Tham số YOLO
    confidence_threshold: float = 0.25      # Ngưỡng độ tin cậy cho detection
    iou_threshold: float = 0.45             # Ngưỡng IoU cho NMS
    max_detections: int = 100               # Số lượng detection tối đa
    
    # Cấu hình tiền xử lý 
    input_size: Tuple[int, int] = (640, 640)  # Kích thước đầu vào cho YOLO
    auto_orient: bool = True                  # Tự động xoay hình ảnh theo EXIF
    
    # Cache
    enable_cache: bool = True                # Bật cache cho kết quả
    cache_expiry: int = 3600                 # Thời gian hết hạn cache (giây)
    
    # Tên class duy nhất
    class_name: str = "medical_object"       # Tên của class duy nhất


class DetectorAgent(BaseAgent):
    """
    Agent phát hiện đối tượng trong hình ảnh y tế sử dụng YOLOv8 với một class duy nhất.
    
    Đặc điểm:
    - Phát hiện đối tượng trong hình ảnh y tế
    - Tối ưu hóa cho một class duy nhất
    - Cung cấp thông tin chi tiết về vị trí và độ tin cậy của đối tượng
    """
    
    def __init__(self, config: DetectorAgentConfig):
        """
        Khởi tạo Detector Agent.
        
        Args:
            config: Cấu hình cho agent
        """
        super().__init__(config)
        self.detector_config = config
        self.device = torch.device(config.device)
        
        # YOLO model
        self.model = None
        
        # Cache kết quả
        self.result_cache = {}
        
        # Metrics
        self.metrics.update({
            "total_detections": 0,
            "avg_detections_per_image": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_confidence": 0,
            "total_confidence": 0
        })
        
        self.logger.info(f"Detector Agent đã được khởi tạo với cấu hình: {self.detector_config}")
    
    def initialize(self) -> bool:
        """
        Khởi tạo Detector Agent, tải model YOLOv8.
        
        Returns:
            bool: True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            self.logger.info(f"Đang tải model YOLOv8 từ {self.detector_config.model_path}")
            
            # Kiểm tra đường dẫn model
            if not os.path.exists(self.detector_config.model_path) and not self.detector_config.model_path.startswith(("http://", "https://")):
                self.logger.warning(f"Không tìm thấy model tại {self.detector_config.model_path}. Kiểm tra lại đường dẫn.")
            
            # Import Ultralytics
            try:
                from ultralytics import YOLO
                
                # Tải model
                self.model = YOLO(self.detector_config.model_path)
                
                # Chuyển model sang device thích hợp
                self.model.to(self.device)
                
                self.logger.info(f"Model YOLOv8 đã được tải thành công trên thiết bị {self.device}")
                self.initialized = True
                return True
                
            except ImportError as e:
                self.logger.error(f"Không thể import ultralytics: {str(e)}")
                self.logger.error("Vui lòng cài đặt ultralytics: pip install ultralytics")
                return False
                
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Detector Agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý yêu cầu phát hiện đối tượng.
        
        Args:
            input_data: Dictionary chứa:
                - image_path: Đường dẫn đến hình ảnh hoặc URL
                - confidence (tùy chọn): Ghi đè ngưỡng độ tin cậy
                - iou (tùy chọn): Ghi đè ngưỡng IoU
                
        Returns:
            Dictionary chứa kết quả và metadata
        """
        if not self.initialized:
            return {"error": "Detector Agent chưa được khởi tạo", "success": False}
        
        # Trích xuất dữ liệu đầu vào
        image_path = input_data.get("image_path")
        
        # Kiểm tra đầu vào
        if not image_path:
            return {"error": "Thiếu đường dẫn hình ảnh (image_path)", "success": False}
        
        # Tạo cache key nếu caching được bật
        cache_key = None
        if self.detector_config.enable_cache:
            # Tạo cache key kết hợp với các tham số detection
            confidence = input_data.get("confidence", self.detector_config.confidence_threshold)
            iou = input_data.get("iou", self.detector_config.iou_threshold)
            
            cache_key = f"{image_path}:{confidence}:{iou}"
            cached_result = self._check_cache(cache_key)
            
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
            
            self.metrics["cache_misses"] += 1
        
        # Thực hiện quá trình xử lý với monitoring
        result = self.run_with_monitoring(self._process_detection, image_path, input_data)
        
        # Lưu vào cache nếu cần
        if self.detector_config.enable_cache and cache_key and "error" not in result:
            self._add_to_cache(cache_key, result)
        
        # Lưu vào bộ nhớ ngắn hạn
        self.remember("last_detection_result", result)
        self.remember(f"detection_result_{image_path}", result)
        
        return result
    
    def _process_detection(self, image_path: Union[str, Image.Image], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý nội bộ cho yêu cầu phát hiện đối tượng.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            input_data: Tham số bổ sung
            
        Returns:
            Dictionary chứa kết quả phát hiện và metadata
        """
        try:
            # Lấy tham số từ input_data hoặc sử dụng giá trị mặc định
            confidence_threshold = input_data.get("confidence", self.detector_config.confidence_threshold)
            iou_threshold = input_data.get("iou", self.detector_config.iou_threshold)
            
            # Tải hình ảnh
            image, original_size = self._load_image(image_path)
            
            # Thực hiện detection
            self.logger.info(f"Đang thực hiện detection trên hình ảnh")
            results = self.model.predict(
                source=image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Xử lý kết quả
            detections = self._process_results(results, original_size)
            
            # Cập nhật metrics
            self.metrics["total_detections"] += len(detections)
            self.metrics["avg_detections_per_image"] = (
                self.metrics["total_detections"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            )
            
            # Tính toán độ tin cậy trung bình
            if detections:
                avg_conf = sum(det["confidence"] for det in detections) / len(detections)
                self.metrics["total_confidence"] += avg_conf
                self.metrics["avg_confidence"] = (
                    self.metrics["total_confidence"] / self.metrics["total_requests"]
                    if self.metrics["total_requests"] > 0 else 0
                )
            
            return {
                "objects": detections,
                "image_path": image_path if isinstance(image_path, str) else "in-memory-image",
                "original_size": original_size,
                "count": len(detections),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý detection: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": f"Không thể xử lý yêu cầu: {str(e)}",
                "success": False
            }
    
    def _load_image(self, image_path: Union[str, Image.Image]) -> Tuple[Union[np.ndarray, Image.Image], Tuple[int, int]]:
        """
        Tải hình ảnh từ đường dẫn hoặc URL.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            
        Returns:
            Tuple: (hình ảnh đã tải, kích thước gốc)
        """
        # Xử lý các loại đầu vào khác nhau
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                # Tải hình ảnh từ URL
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                # Tải hình ảnh từ đường dẫn cục bộ
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Không tìm thấy file hình ảnh: {image_path}")
                image = Image.open(image_path)
        else:
            # Đã là PIL Image
            image = image_path
        
        # Tự động xoay hình ảnh theo EXIF nếu cần
        if self.detector_config.auto_orient and hasattr(image, '_getexif'):
            try:
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274, 1)  # 274 là tag cho orientation
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            except Exception as e:
                self.logger.warning(f"Không thể xử lý EXIF orientation: {str(e)}")
        
        # Lưu kích thước gốc
        original_size = image.size
        
        return image, original_size
    
    def _process_results(self, results, original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Xử lý kết quả từ model YOLOv8.
        
        Args:
            results: Kết quả từ model.predict()
            original_size: Kích thước gốc của hình ảnh
            
        Returns:
            List[Dict]: Danh sách các đối tượng phát hiện được
        """
        detections = []
        
        for i, result in enumerate(results):
            # Lấy thông tin boxes
            boxes = result.boxes
            
            for j, box in enumerate(boxes):
                # Lấy các thông tin từ box
                xyxy = box.xyxy[0].cpu().numpy()  # Tọa độ [x1, y1, x2, y2]
                conf = float(box.conf[0].item())  # Độ tin cậy
                
                # Tính các thông số bổ sung
                x1, y1, x2, y2 = xyxy.tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = width * height
                
                # Tính tỷ lệ so với hình ảnh gốc
                img_width, img_height = original_size
                area_ratio = area / (img_width * img_height) if img_width * img_height > 0 else 0
                
                detection = {
                    "id": j,
                    "bbox": xyxy.tolist(),
                    "confidence": conf,
                    "class": self.detector_config.class_name,
                    "width": width,
                    "height": height,
                    "center": [center_x, center_y],
                    "area": area,
                    "area_ratio": area_ratio
                }
                
                detections.append(detection)
        
        # Sắp xếp theo độ tin cậy giảm dần
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        # Giới hạn số lượng detection
        if len(detections) > self.detector_config.max_detections:
            detections = detections[:self.detector_config.max_detections]
        
        # Đánh lại id sau khi sắp xếp
        for i, det in enumerate(detections):
            det["id"] = i
        
        return detections
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Kiểm tra cache cho kết quả đã tính toán trước đó."""
        if not self.detector_config.enable_cache:
            return None
            
        cached_item = self.result_cache.get(key)
        if not cached_item:
            return None
            
        # Kiểm tra thời gian hết hạn
        if time.time() - cached_item["timestamp"] > self.detector_config.cache_expiry:
            # Cache đã hết hạn
            del self.result_cache[key]
            return None
            
        self.logger.info(f"Đã tìm thấy kết quả trong cache cho key: {key}")
        return cached_item["result"]
    
    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """Thêm kết quả vào cache."""
        if not self.detector_config.enable_cache:
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
        Xử lý hàng loạt các yêu cầu phát hiện đối tượng.
        
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