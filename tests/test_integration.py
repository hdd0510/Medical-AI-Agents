#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Tests for Medical AI System
--------------------------------------
Kiểm thử tích hợp toàn bộ hệ thống AI y tế đa agent.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import numpy as np
from PIL import Image
import io
import pytest
import tempfile
import shutil

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import orchestrator và các agent để test
from orchestrator.main import MedicalOrchestrator, OrchestratorConfig
from agents.detector import MedicalDetectorAgent, DetectorConfig
from agents.classifier import MedicalClassifierAgent, ClassifierConfig
from agents.vqa import MedicalVQAAgent, VQAAgentConfig
from agents.rag import MedicalRAGAgent, RAGConfig

# Tạo ảnh test giả lập
def create_test_image(width=224, height=224):
    """Tạo ảnh test cho việc kiểm thử."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    return img_pil


class IntegrationTestConfig:
    """Cấu hình cho tests tích hợp."""
    
    # Đường dẫn test
    TEST_DATA_DIR = os.path.join('tests', 'test_data')
    TEST_OUTPUT_DIR = os.path.join('tests', 'test_output')
    
    # Cấu hình model weights (dummy paths cho testing)
    DETECTOR_MODEL_PATH = os.path.join('weights', 'detector', 'model.pt')
    CLASSIFIER1_MODEL_PATH = os.path.join('weights', 'classifier_1', 'model.pt')
    CLASSIFIER2_MODEL_PATH = os.path.join('weights', 'classifier_2', 'model.pt')
    VQA_MODEL_PATH = os.path.join('weights', 'vqa', 'model.pt')
    RAG_MODEL_PATH = os.path.join('weights', 'rag', 'model.pt')
    
    # Cấu hình class names
    CLASSIFIER1_CLASSES = ["WLI", "BLI", "LCI"]
    CLASSIFIER2_CLASSES = ["Esophagus", "Stomach", "Duodenum", "Colon"]
    
    # Test queries
    TEST_QUERIES = [
        "Is there a polyp in this image?",
        "What part of GI tract is shown?",
        "What imaging modality is used?",
        "Describe what you see in this image."
    ]


@pytest.fixture(scope="module")
def setup_test_dirs():
    """Thiết lập thư mục test."""
    # Tạo thư mục test data nếu chưa tồn tại
    os.makedirs(IntegrationTestConfig.TEST_DATA_DIR, exist_ok=True)
    os.makedirs(IntegrationTestConfig.TEST_OUTPUT_DIR, exist_ok=True)
    
    # Tạo và lưu một số ảnh test
    for i in range(5):
        img = create_test_image()
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, f"test_image_{i}.jpg")
        img.save(img_path)
    
    yield
    
    # Dọn dẹp sau khi test xong
    shutil.rmtree(IntegrationTestConfig.TEST_OUTPUT_DIR)


@pytest.fixture(scope="module")
def mock_all_agents():
    """Mock tất cả các agents cho testing."""
    with patch('agents.detector.MedicalDetectorAgent._load_model'), \
         patch('agents.classifier.MedicalClassifierAgent._load_model'), \
         patch('agents.classifier_2.MedicalClassifierAgent2._load_model'), \
         patch('agents.vqa.MedicalVQAAgent._load_model'), \
         patch('agents.rag.MedicalRAGAgent._load_model'):
        
        # Tạo config cho các agents
        detector_config = DetectorConfig(
            model_path=IntegrationTestConfig.DETECTOR_MODEL_PATH,
            device="cpu",
            confidence_threshold=0.5
        )
        
        classifier_config = ClassifierConfig(
            model_path="test_model.pt",
            class_names=["class1", "class2"],
            device="cpu"
        )
        
        classifier2_config = ClassifierConfig2(
            model_path=IntegrationTestConfig.CLASSIFIER2_MODEL_PATH,
            device="cpu",
            class_names=IntegrationTestConfig.CLASSIFIER2_CLASSES
        )
        
        vqa_config = VQAAgentConfig(
            model_path=IntegrationTestConfig.VQA_MODEL_PATH,
            device="cpu",
            max_tokens=100
        )
        
        rag_config = RAGConfig(
            model_path=IntegrationTestConfig.RAG_MODEL_PATH,
            device="cpu",
            knowledge_base_path="data/medical_knowledge"
        )
        
        # Tạo các agents
        detector_agent = MedicalDetectorAgent(detector_config)
        classifier_agent = MedicalClassifierAgent(classifier_config)
        classifier2_agent = MedicalClassifierAgent2(classifier2_config)
        vqa_agent = MedicalVQAAgent(vqa_config)
        rag_agent = MedicalRAGAgent(rag_config)
        
        # Mock các methods
        detector_agent.process = MagicMock(return_value={
            'success': True,
            'detections': [{'box': [10, 20, 30, 40], 'confidence': 0.95, 'class_id': 1, 'class_name': 'polyp'}],
            'detection_count': 1
        })
        
        classifier_agent.process = MagicMock(return_value={
            'success': True,
            'class_id': 1,
            'class_name': 'BLI',
            'confidence': 0.85
        })
        
        classifier2_agent.process = MagicMock(return_value={
            'success': True,
            'class_id': 3,
            'class_name': 'Colon',
            'confidence': 0.9
        })
        
        vqa_agent.process = MagicMock(return_value={
            'success': True,
            'answer': 'The image shows a polyp in the colon using BLI imaging.',
            'confidence': 0.8
        })
        
        rag_agent.process = MagicMock(return_value={
            'success': True,
            'answer': 'Polyps are abnormal tissue growths that can be precancerous.',
            'sources': ['medical_textbook_1'],
            'confidence': 0.9
        })
        
        return {
            "detector": detector_agent,
            "classifier": classifier_agent,
            "classifier_2": classifier2_agent,
            "vqa": vqa_agent,
            "rag": rag_agent
        }


@pytest.fixture(scope="module")
def orchestrator_with_mocked_agents(mock_all_agents):
    """Tạo orchestrator với các agents đã được mock."""
    # Tạo config cho orchestrator
    config = OrchestratorConfig(
        device="cpu",
        parallel_execution=True,
        use_reflection=True,
        output_path=IntegrationTestConfig.TEST_OUTPUT_DIR
    )
    
    # Khởi tạo orchestrator
    with patch('memory.conversation_memory.ConversationMemory'), \
         patch('memory.vector_store.VectorStore'):
        orchestrator = MedicalOrchestrator(config)
        
        # Đăng ký các agents
        for name, agent in mock_all_agents.items():
            orchestrator.register_agent(name, agent)
            
        # Mock LLM client cho reflection
        orchestrator.llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "has_bias": False,
            "bias_explanation": "",
            "confidence_assessment": "High confidence",
            "consistency_assessment": "Consistent with image analysis",
            "improved_answer": "The image shows a polyp in the colon visualized with BLI imaging.",
            "reasoning": "Combined information from detection and classification."
        })
        orchestrator.llm_client.chat.completions.create.return_value = mock_response
        
        return orchestrator


class TestFullWorkflow:
    """Test case cho workflow đầy đủ của hệ thống."""
    
    def test_sequential_flow(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra luồng xử lý tuần tự."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Đặt parallel_execution = False
        orchestrator.config.parallel_execution = False
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_0.jpg")
        
        # Thực thi orchestrate
        result = orchestrator.orchestrate(img_path, IntegrationTestConfig.TEST_QUERIES[0])
        
        # Kiểm tra kết quả
        assert "answer" in result
        assert "metadata" in result
        assert "session_id" in result["metadata"]
        assert os.path.exists(os.path.join(IntegrationTestConfig.TEST_OUTPUT_DIR, result["metadata"]["session_id"], "result.json"))
    
    def test_parallel_flow(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra luồng xử lý song song."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Đặt parallel_execution = True
        orchestrator.config.parallel_execution = True
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_1.jpg")
        
        # Thực thi orchestrate với comprehensive task
        result = orchestrator.orchestrate(img_path)  # Không có query -> comprehensive task
        
        # Kiểm tra kết quả
        assert "modality" in result
        assert "region" in result
        assert "detections" in result
        assert "summary" in result
    
    def test_medical_qa_flow(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra luồng xử lý medical QA."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_2.jpg")
        
        # Override _analyze_request để luôn trả về medical_qa
        orchestrator._analyze_request = MagicMock(return_value=("medical_qa", [
            {"agent": "vqa", "params": {"query": IntegrationTestConfig.TEST_QUERIES[0]}}
        ]))
        
        # Thực thi orchestrate
        result = orchestrator.orchestrate(img_path, IntegrationTestConfig.TEST_QUERIES[0])
        
        # Kiểm tra kết quả
        assert "answer" in result
        assert result["answer"] == "The image shows a polyp in the colon using BLI imaging."
    
    def test_polyp_detection_flow(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra luồng xử lý phát hiện polyp."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_3.jpg")
        
        # Override _analyze_request để luôn trả về polyp_detection
        orchestrator._analyze_request = MagicMock(return_value=("polyp_detection", [
            {"agent": "detector", "params": {}}
        ]))
        
        # Thực thi orchestrate
        result = orchestrator.orchestrate(img_path, "Are there any polyps?")
        
        # Kiểm tra kết quả
        assert "detections" in result
        assert result["detection_count"] == 1
    
    def test_reflection_flow(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra luồng reflection."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_4.jpg")
        
        # Override các method liên quan đến reflection
        orchestrator._needs_reflection = MagicMock(return_value=True)
        
        # Override _analyze_request để bao gồm VQA
        orchestrator._analyze_request = MagicMock(return_value=("medical_qa", [
            {"agent": "detector", "params": {}},
            {"agent": "vqa", "params": {"query": IntegrationTestConfig.TEST_QUERIES[3]}}
        ]))
        
        # Thực thi orchestrate
        result = orchestrator.orchestrate(img_path, IntegrationTestConfig.TEST_QUERIES[3])
        
        # Kiểm tra kết quả reflection
        assert "reflection" in result
        assert "improved_answer" in result["reflection"]
    
    def test_error_handling(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra xử lý lỗi trong quá trình xử lý."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Lấy đường dẫn ảnh test không tồn tại
        img_path = "non_existent_image.jpg"
        
        # Thực thi orchestrate
        result = orchestrator.orchestrate(img_path, IntegrationTestConfig.TEST_QUERIES[0])
        
        # Kiểm tra xử lý lỗi
        assert "error" in result
        assert result["success"] is False
    
    def test_medical_context(self, setup_test_dirs, orchestrator_with_mocked_agents):
        """Kiểm tra sử dụng thông tin y tế bổ sung."""
        # Lấy orchestrator
        orchestrator = orchestrator_with_mocked_agents
        
        # Lấy đường dẫn ảnh test
        img_path = os.path.join(IntegrationTestConfig.TEST_DATA_DIR, "test_image_0.jpg")
        
        # Tạo medical context
        medical_context = {
            "patient_history": "Family history of colon cancer",
            "previous_findings": "Previous colonoscopy showed small polyps",
            "patient_age": 55
        }
        
        # Thực thi orchestrate với medical context
        result = orchestrator.orchestrate(img_path, IntegrationTestConfig.TEST_QUERIES[0], medical_context)
        
        # Kiểm tra kết quả
        assert "answer" in result
        assert "metadata" in result
        assert "medical_context" in result


if __name__ == '__main__':
    pytest.main(['-v', 'test_integration.py']) 