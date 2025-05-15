#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Suite for Medical AI System Orchestrator
--------------------------------------------
Kiểm thử Orchestrator trong hệ thống AI y tế đa agent.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
import json
import numpy as np
from PIL import Image
import io
import pytest
import tempfile
import time

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import orchestrator để test
from orchestrator.main import MedicalOrchestrator, OrchestratorConfig
from agents.detector import MedicalDetectorAgent, DetectorConfig
from agents.classifier_1 import MedicalClassifierAgent1, ClassifierConfig1
from agents.classifier_2 import MedicalClassifierAgent2, ClassifierConfig2
from agents.vqa import MedicalVQAAgent, VQAAgentConfig
from agents.rag import MedicalRAGAgent, RAGConfig

# Tạo ảnh test giả lập
def create_test_image(width=224, height=224):
    """Tạo ảnh test cho việc kiểm thử."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    return img_pil


class OrchestratorConfigTest(unittest.TestCase):
    """Test case cho OrchestratorConfig."""
    
    def test_default_config(self):
        """Kiểm tra config mặc định."""
        config = OrchestratorConfig()
        self.assertEqual(config.name, "Medical Orchestrator")
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.parallel_execution)
        self.assertTrue(config.use_reflection)
        self.assertGreater(config.consistency_threshold, 0.5)


class OrchestratorInitTest(unittest.TestCase):
    """Test case cho khởi tạo MedicalOrchestrator."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(device="cpu", parallel_execution=False)
        
        # Mock memory và LLM
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'), \
             patch('orchestrator.main.MedicalOrchestrator._initialize_llm'):
            self.orchestrator = MedicalOrchestrator(self.config)
    
    def test_orchestrator_initialization(self):
        """Kiểm tra khởi tạo của MedicalOrchestrator."""
        self.assertEqual(self.orchestrator.config.name, "Medical Orchestrator")
        self.assertEqual(self.orchestrator.config.device, "cpu")
        self.assertFalse(self.orchestrator.config.parallel_execution)
        self.assertEqual(len(self.orchestrator.agents), 0)
    
    def test_register_agent(self):
        """Kiểm tra đăng ký agent."""
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        
        self.orchestrator.register_agent("mock_agent", mock_agent)
        self.assertIn("mock_agent", self.orchestrator.agents)
        self.assertEqual(self.orchestrator.agents["mock_agent"], mock_agent)


class OrchestratorAnalyzeRequestTest(unittest.TestCase):
    """Test case cho phân tích yêu cầu."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(device="cpu")
        
        # Mock memory và LLM
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'), \
             patch('orchestrator.main.MedicalOrchestrator._initialize_llm'):
            self.orchestrator = MedicalOrchestrator(self.config)
            self.orchestrator.llm_client = MagicMock()
    
    def test_analyze_request_with_llm(self):
        """Kiểm tra phân tích yêu cầu với LLM."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "medical_qa"
        self.orchestrator.llm_client.chat.completions.create.return_value = mock_response
        
        query = "Is there a polyp in this image?"
        task_type, execution_plan = self.orchestrator._analyze_request(query, None)
        
        # Kiểm tra kết quả
        self.assertEqual(task_type, "medical_qa")
        self.assertIsInstance(execution_plan, list)
        
        # Verify LLM được gọi
        self.orchestrator.llm_client.chat.completions.create.assert_called_once()
    
    def test_analyze_request_without_query(self):
        """Kiểm tra phân tích yêu cầu khi không có query."""
        task_type, execution_plan = self.orchestrator._analyze_request(None, None)
        
        # Kiểm tra kết quả
        self.assertEqual(task_type, "comprehensive")
        self.assertIsInstance(execution_plan, list)
        self.assertGreaterEqual(len(execution_plan), 2)  # Ít nhất 2 bước
    
    def test_analyze_request_fallback(self):
        """Kiểm tra fallback khi LLM fails."""
        # Mock LLM để raise exception
        self.orchestrator.llm_client.chat.completions.create.side_effect = Exception("API Error")
        
        query = "What is visible in this image?"
        task_type, execution_plan = self.orchestrator._analyze_request(query, None)
        
        # Kiểm tra fallback đến simple classification
        self.assertIn(task_type, self.orchestrator.config.task_types)
        self.assertIsInstance(execution_plan, list)
        self.assertGreaterEqual(len(execution_plan), 1)


class OrchestratorExecutionTest(unittest.TestCase):
    """Test case cho execution steps."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(device="cpu")
        
        # Mock memory và LLM
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'), \
             patch('orchestrator.main.MedicalOrchestrator._initialize_llm'):
            self.orchestrator = MedicalOrchestrator(self.config)
        
        # Đăng ký mock agents
        self.detector_agent = MagicMock()
        self.detector_agent.process.return_value = {
            'success': True,
            'detections': [{'box': [10, 20, 30, 40], 'confidence': 0.95, 'class_id': 1, 'class_name': 'polyp'}],
            'detection_count': 1
        }
        self.orchestrator.register_agent("detector", self.detector_agent)
        
        self.classifier1_agent = MagicMock()
        self.classifier1_agent.process.return_value = {
            'success': True,
            'class_id': 1,
            'class_name': 'BLI',
            'confidence': 0.85
        }
        self.orchestrator.register_agent("classifier_1", self.classifier1_agent)
        
        self.vqa_agent = MagicMock()
        self.vqa_agent.process.return_value = {
            'success': True,
            'answer': 'The image shows a polyp in the colon.',
            'confidence': 0.8
        }
        self.orchestrator.register_agent("vqa", self.vqa_agent)
    
    def test_execute_step(self):
        """Kiểm tra thực thi từng bước."""
        # Tạo ảnh test
        test_img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            test_img.save(tmp.name)
            
            # Thực thi detector
            step = {"agent": "detector", "params": {}}
            result = self.orchestrator._execute_step(step, tmp.name, None, None, {})
            
            # Kiểm tra kết quả
            self.assertIn("detector", result)
            self.assertTrue(result["detector"]["success"])
            self.assertEqual(result["detector"]["detection_count"], 1)
            
            # Verify detector agent được gọi
            self.detector_agent.process.assert_called_once_with(tmp.name)
    
    def test_execute_step_with_params(self):
        """Kiểm tra thực thi bước với params."""
        # Tạo ảnh test
        test_img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            test_img.save(tmp.name)
            
            # Thực thi VQA với query
            query = "What is in this image?"
            step = {"agent": "vqa", "params": {"query": query}}
            result = self.orchestrator._execute_step(step, tmp.name, query, None, {})
            
            # Kiểm tra kết quả
            self.assertIn("vqa", result)
            self.assertTrue(result["vqa"]["success"])
            self.assertEqual(result["vqa"]["answer"], "The image shows a polyp in the colon.")
            
            # Verify VQA agent được gọi với đúng params
            self.vqa_agent.process.assert_called_once_with(tmp.name, query=query)
    
    def test_execute_perception_agents_parallel(self):
        """Kiểm tra thực thi song song các agents."""
        # Tạo ảnh test
        test_img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            test_img.save(tmp.name)
            
            # Thực thi parallel
            results = self.orchestrator._execute_perception_agents_parallel(tmp.name)
            
            # Kiểm tra kết quả
            self.assertIn("detector", results)
            self.assertIn("classifier_1", results)
            
            # Verify cả hai agents được gọi
            self.detector_agent.process.assert_called_once_with(tmp.name)
            self.classifier1_agent.process.assert_called_once_with(tmp.name)


class OrchestratorReflectionTest(unittest.TestCase):
    """Test case cho reflection trong MedicalOrchestrator."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(
            device="cpu",
            use_reflection=True,
            consistency_threshold=0.7
        )
        
        # Mock memory và LLM
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'):
            self.orchestrator = MedicalOrchestrator(self.config)
            # Mock LLM client
            self.orchestrator.llm_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = json.dumps({
                "has_bias": True,
                "bias_explanation": "The answer assumes without evidence.",
                "confidence_assessment": "Medium confidence",
                "consistency_assessment": "Consistent with image analysis",
                "improved_answer": "The image likely shows a polyp in the colon, but additional tests are recommended.",
                "reasoning": "Added caution and recommendation for further testing."
            })
            self.orchestrator.llm_client.chat.completions.create.return_value = mock_response
            
            # Mock vector store
            self.orchestrator.vector_store.search.return_value = [
                {"query": "Is this a polyp?", "answer": "Yes, it appears to be a small polyp.", "score": 0.9}
            ]
    
    def test_needs_reflection(self):
        """Kiểm tra logic xác định cần reflection hay không."""
        # Trường hợp 1: Confidence thấp
        vqa_result_low_conf = {"answer": "This might be a polyp.", "confidence": 0.5}
        self.assertTrue(self.orchestrator._needs_reflection("Is this a polyp?", vqa_result_low_conf))
        
        # Trường hợp 2: Confidence cao
        vqa_result_high_conf = {"answer": "This is definitely a polyp.", "confidence": 0.95}
        self.assertFalse(self.orchestrator._needs_reflection("Is this a polyp?", vqa_result_high_conf))
    
    def test_find_similar_questions(self):
        """Kiểm tra tìm câu hỏi tương tự."""
        similar_questions = self.orchestrator._find_similar_questions("Is this a polyp?")
        
        # Kiểm tra kết quả
        self.assertIsInstance(similar_questions, list)
        self.assertEqual(len(similar_questions), 1)
        self.assertIn("query", similar_questions[0])
        self.assertIn("answer", similar_questions[0])
        
        # Verify vector store được gọi
        self.orchestrator.vector_store.search.assert_called_once()
    
    def test_perform_reflection(self):
        """Kiểm tra thực hiện reflection."""
        query = "Is this a polyp?"
        vqa_result = {"answer": "This is a polyp in the colon.", "confidence": 0.6}
        other_results = {
            "detector": {"detections": [{"class_name": "polyp", "confidence": 0.95}]},
            "classifier_2": {"class_name": "Colon", "confidence": 0.9}
        }
        
        reflection_result = self.orchestrator._perform_reflection(query, vqa_result, None, other_results)
        
        # Kiểm tra kết quả
        self.assertIn("has_bias", reflection_result)
        self.assertTrue(reflection_result["has_bias"])
        self.assertIn("improved_answer", reflection_result)
        self.assertIn("confidence_assessment", reflection_result)
        
        # Verify LLM được gọi
        self.orchestrator.llm_client.chat.completions.create.assert_called_once()


class OrchestratorMainWorkflowTest(unittest.TestCase):
    """Test case cho workflow chính của MedicalOrchestrator."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(
            device="cpu",
            parallel_execution=False,
            use_reflection=True,
            output_path="test_results"
        )
        
        # Khởi tạo orchestrator với mocks
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'), \
             patch('orchestrator.main.MedicalOrchestrator._initialize_llm'), \
             patch('orchestrator.main.MedicalOrchestrator._analyze_request'), \
             patch('orchestrator.main.MedicalOrchestrator._execute_step'), \
             patch('orchestrator.main.MedicalOrchestrator._needs_reflection'), \
             patch('orchestrator.main.MedicalOrchestrator._perform_reflection'), \
             patch('orchestrator.main.MedicalOrchestrator._synthesize_results'), \
             patch('orchestrator.main.MedicalOrchestrator._store_in_memory'), \
             patch('os.makedirs'):
            self.orchestrator = MedicalOrchestrator(self.config)
            
            # Mock các method chính
            self.orchestrator._analyze_request.return_value = ("medical_qa", [
                {"agent": "detector", "params": {}},
                {"agent": "vqa", "params": {"query": "Is this a polyp?"}}
            ])
            
            def mock_execute_step(step, image_path, query, medical_context, previous_results):
                if step["agent"] == "detector":
                    return {"detector": {"success": True, "detection_count": 1}}
                elif step["agent"] == "vqa":
                    return {"vqa": {"success": True, "answer": "Yes, it's a polyp.", "confidence": 0.7}}
                return {}
            
            self.orchestrator._execute_step.side_effect = mock_execute_step
            self.orchestrator._needs_reflection.return_value = True
            self.orchestrator._perform_reflection.return_value = {
                "has_bias": False,
                "improved_answer": "Yes, it appears to be a polyp with high confidence."
            }
            self.orchestrator._synthesize_results.return_value = {
                "answer": "Yes, it appears to be a polyp with high confidence.",
                "detections": {"count": 1}
            }
    
    def test_orchestrate_workflow(self):
        """Kiểm tra workflow chính của orchestrate."""
        # Tạo ảnh test
        test_img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            test_img.save(tmp.name)
            
            # Thực thi orchestrate
            query = "Is this a polyp?"
            result = self.orchestrator.orchestrate(tmp.name, query)
            
            # Kiểm tra workflow
            self.orchestrator._analyze_request.assert_called_once_with(query, None)
            self.assertEqual(self.orchestrator._execute_step.call_count, 2)
            self.orchestrator._needs_reflection.assert_called_once()
            self.orchestrator._perform_reflection.assert_called_once()
            self.orchestrator._synthesize_results.assert_called_once()
            self.orchestrator._store_in_memory.assert_called_once()
            
            # Kiểm tra kết quả
            self.assertIn("answer", result)
            self.assertIn("metadata", result)
            self.assertIn("session_id", result["metadata"])
    
    def test_orchestrate_with_error(self):
        """Kiểm tra xử lý lỗi trong orchestrate."""
        # Tạo ảnh test
        test_img = create_test_image()
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            test_img.save(tmp.name)
            
            # Giả lập lỗi
            self.orchestrator._analyze_request.side_effect = Exception("Test error")
            
            # Thực thi orchestrate
            result = self.orchestrator.orchestrate(tmp.name, "Is this a polyp?")
            
            # Kiểm tra xử lý lỗi
            self.assertIn("error", result)
            self.assertIn("session_id", result)
            self.assertFalse(result["success"])


class OrchestratorResultsTest(unittest.TestCase):
    """Test case cho xử lý kết quả của MedicalOrchestrator."""
    
    def setUp(self):
        """Thiết lập cho test."""
        self.config = OrchestratorConfig(device="cpu")
        
        # Mock memory và LLM
        with patch('memory.conversation_memory.ConversationMemory'), \
             patch('memory.vector_store.VectorStore'), \
             patch('orchestrator.main.MedicalOrchestrator._initialize_llm'):
            self.orchestrator = MedicalOrchestrator(self.config)
    
    def test_synthesize_results_medical_qa(self):
        """Kiểm tra tổng hợp kết quả cho medical_qa."""
        results = {
            "detector": {
                "success": True,
                "detections": [{"class_name": "polyp", "confidence": 0.95}],
                "detection_count": 1
            },
            "vqa": {
                "success": True,
                "answer": "This is a small polyp in the colon.",
                "confidence": 0.85
            }
        }
        
        final_result = self.orchestrator._synthesize_results("medical_qa", results, "What is in this image?")
        
        # Kiểm tra kết quả
        self.assertIn("answer", final_result)
        self.assertEqual(final_result["answer"], "This is a small polyp in the colon.")
        self.assertIn("confidence", final_result)
        self.assertIn("detection_details", final_result)
        self.assertIn("summary", final_result)
    
    def test_synthesize_results_comprehensive(self):
        """Kiểm tra tổng hợp kết quả cho comprehensive."""
        results = {
            "detector": {
                "success": True,
                "detections": [{"class_name": "polyp", "confidence": 0.95}],
                "detection_count": 1
            },
            "classifier_1": {
                "success": True,
                "class_name": "BLI",
                "confidence": 0.85
            },
            "classifier_2": {
                "success": True,
                "class_name": "Colon",
                "confidence": 0.9
            }
        }
        
        final_result = self.orchestrator._synthesize_results("comprehensive", results, None)
        
        # Kiểm tra kết quả
        self.assertIn("detections", final_result)
        self.assertIn("modality", final_result)
        self.assertIn("region", final_result)
        self.assertIn("summary", final_result)


if __name__ == '__main__':
    unittest.main() 