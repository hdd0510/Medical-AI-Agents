#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Orchestrator
-------------------------------
Orchestrator trung tâm cho hệ thống AI y tế đa agent với các chức năng phân tích hình ảnh nội soi
tiêu hóa, phát hiện polyp và trả lời câu hỏi y tế với reflection để tránh bias.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import traceback
import uuid

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import các agent
from agents.detector import DetectorAgent, DetectorAgentConfig
from agents.classifier_1 import MedicalClassifierAgent1, ClassifierConfig1
from agents.classifier_2 import MedicalClassifierAgent2, ClassifierConfig2
from agents.vqa import MedicalVQAAgent, VQAAgentConfig
# from agents.rag import MedicalRAGAgent, RAGConfig  # RAG Agent tạm thời bị vô hiệu hóa do lỗi thư viện

try:
    from agents.rag import MedicalRAGAgent, RAGConfig
except ImportError:
    # Tạo class giả để tránh lỗi
    class RAGConfig:
        pass
    class MedicalRAGAgent:
        pass

# Import LLM client cho reflection
from memory.vector_store import VectorStore
from memory.conversation_memory import ConversationMemory

@dataclass
class OrchestratorConfig:
    """Cấu hình cho Medical Orchestrator."""
    # Cấu hình chung
    name: str = "Medical Orchestrator"
    device: str = "cuda"
    logging_level: str = "INFO"
    
    # Cấu hình workflow
    parallel_execution: bool = True  # Chế độ xử lý đồng thời hoặc tuần tự
    use_reflection: bool = True      # Bật/tắt reflection
    consistency_threshold: float = 0.7  # Ngưỡng phát hiện bias
    similarity_threshold: float = 0.75  # Ngưỡng tương đồng câu hỏi
    
    # Cấu hình LLM cho reflection và điều phối
    llm_api_key: str = None
    llm_model: str = "gpt-4"
    
    # Cấu hình memory
    memory_enabled: bool = True
    memory_max_entries: int = 100
    
    # Thông số cho các loại yêu cầu
    task_types: List[str] = field(default_factory=lambda: [
        "polyp_detection", "modality_classification", 
        "region_classification", "medical_qa", "comprehensive"
    ])
    
    # Đường dẫn lưu kết quả
    output_path: str = "results"


class MedicalOrchestrator:
    """
    Orchestrator trung tâm cho hệ thống AI y tế đa agent.
    
    Chịu trách nhiệm:
    1. Phân tích yêu cầu từ người dùng
    2. Lập kế hoạch thực thi sử dụng các agents
    3. Điều phối việc thực thi các agents
    4. Phát hiện và xử lý bias trong LLaVA thông qua reflection
    5. Tổng hợp kết quả từ các agents thành phản hồi cuối cùng
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Khởi tạo Medical Orchestrator."""
        self.config = config
        self.logger = logging.getLogger(f"orchestrator.{self.config.name}")
        self.logger.setLevel(getattr(logging, config.logging_level))
        
        # Khởi tạo các agents
        self.agents = {}
        
        # Khởi tạo memory
        if self.config.memory_enabled:
            self.conversation_memory = ConversationMemory(max_entries=self.config.memory_max_entries)
            self.vector_store = VectorStore()
        
        # LLM client cho reflection và phân tích
        self.llm_client = self._initialize_llm()
        
        # Session ID
        self.current_session_id = None
        
        self.logger.info(f"Medical Orchestrator đã được khởi tạo")
    
    def _initialize_llm(self):
        """Khởi tạo LLM client cho reflection và phân tích."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.config.llm_api_key)
            return client
        except ImportError:
            self.logger.warning("OpenAI không được cài đặt. Reflection có thể bị giới hạn.")
            return None
        except Exception as e:
            self.logger.error(f"Không thể khởi tạo LLM client: {str(e)}")
            return None
    
    def register_agent(self, name: str, agent):
        """Đăng ký một agent với orchestrator."""
        self.agents[name] = agent
        self.logger.info(f"Đã đăng ký agent: {name}")
    
    def orchestrate(self, 
                   image_path: str, 
                   query: Optional[str] = None, 
                   medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Điều phối quá trình xử lý hình ảnh y tế và câu hỏi.
        
        Args:
            image_path: Đường dẫn đến hình ảnh
            query: Câu hỏi hoặc yêu cầu từ người dùng (tùy chọn)
            medical_context: Thông tin y tế bổ sung (tùy chọn)
            
        Returns:
            Dictionary chứa kết quả xử lý
        """
        start_time = time.time()
        
        # Tạo session ID
        self.current_session_id = str(uuid.uuid4())
        
        # Tạo output directory nếu cần
        output_dir = os.path.join(self.config.output_path, self.current_session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.logger.info(f"Bắt đầu xử lý session {self.current_session_id}")
            self.logger.info(f"Hình ảnh: {image_path}")
            self.logger.info(f"Câu hỏi: {query if query else 'Không có'}")
            
            # 1. Phân tích yêu cầu
            task_type, execution_plan = self._analyze_request(query, medical_context)
            
            # 2. Thực thi các agents theo kế hoạch
            results = {}
            
            if self.config.parallel_execution and task_type == "comprehensive":
                # Thực thi song song cho các agents perception
                perception_results = self._execute_perception_agents_parallel(image_path)
                results.update(perception_results)
            else:
                # Thực thi tuần tự theo kế hoạch
                for step in execution_plan:
                    step_result = self._execute_step(step, image_path, query, medical_context, results)
                    results.update(step_result)
            
            # 3. Phát hiện và xử lý bias nếu có
            if query and "vqa" in results and self.config.use_reflection:
                vqa_result = results["vqa"]
                
                # Kiểm tra xem có cần reflection không
                if self._needs_reflection(query, vqa_result):
                    self.logger.info("Phát hiện bias hoặc độ tin cậy thấp. Thực hiện reflection...")
                    reflection_result = self._perform_reflection(query, vqa_result, medical_context, results)
                    results["reflection"] = reflection_result
            
            # 4. Tổng hợp kết quả cuối cùng
            final_result = self._synthesize_results(task_type, results, query)
            
            # 5. Lưu vào memory
            if self.config.memory_enabled:
                self._store_in_memory(image_path, query, results, final_result, medical_context)
            
            # Thêm metadata
            elapsed_time = time.time() - start_time
            final_result["metadata"] = {
                "session_id": self.current_session_id,
                "task_type": task_type,
                "processing_time": elapsed_time,
                "timestamp": time.time(),
                "query": query,
                "image_path": image_path
            }
            
            # Lưu kết quả nếu cần
            result_path = os.path.join(output_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Xử lý hoàn tất. Thời gian: {elapsed_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình xử lý: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                "error": f"Không thể xử lý yêu cầu: {str(e)}",
                "session_id": self.current_session_id,
                "success": False
            }
    
    def _analyze_request(self, query: Optional[str], medical_context: Optional[Dict]) -> Tuple[str, List[Dict]]:
        """
        Phân tích yêu cầu và tạo kế hoạch thực thi.
        
        Args:
            query: Câu hỏi hoặc yêu cầu
            medical_context: Thông tin y tế bổ sung
            
        Returns:
            Tuple chứa loại tác vụ và kế hoạch thực thi
        """
        # Mặc định là phân tích tổng quát nếu không có query
        if not query:
            return "comprehensive", [
                {"agent": "detector", "params": {}},
                {"agent": "classifier_1", "params": {}},
                {"agent": "classifier_2", "params": {}}
            ]
        
        # Sử dụng LLM để phân tích nếu có
        if self.llm_client:
            try:
                prompt = f"""
                Phân tích yêu cầu sau và xác định loại tác vụ phù hợp nhất:
                
                Yêu cầu: {query}
                
                Loại tác vụ có thể là một trong các loại sau:
                - polyp_detection: Phát hiện polyp
                - modality_classification: Phân loại kỹ thuật nội soi (BLI, WLI, ...)
                - region_classification: Phân loại vị trí trong đường tiêu hóa
                - medical_qa: Trả lời câu hỏi y tế về hình ảnh
                - comprehensive: Yêu cầu tổng hợp nhiều loại phân tích
                
                Trả về duy nhất tên loại tác vụ phù hợp nhất (không có giải thích).
                """
                
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                
                task_type = response.choices[0].message.content.strip().lower()
                
                # Chuẩn hóa task_type
                for valid_type in self.config.task_types:
                    if valid_type.lower() in task_type:
                        task_type = valid_type
                        break
                else:
                    # Fallback nếu không khớp
                    task_type = "comprehensive"
                
            except Exception as e:
                self.logger.warning(f"Không thể sử dụng LLM để phân tích: {str(e)}")
                # Fallback: phân tích đơn giản dựa trên từ khóa
                task_type = self._simple_task_classification(query)
        else:
            # Phân tích đơn giản dựa trên từ khóa
            task_type = self._simple_task_classification(query)
        
        # Tạo kế hoạch thực thi dựa trên loại tác vụ
        execution_plan = self._create_execution_plan(task_type, query)
        
        self.logger.info(f"Loại tác vụ: {task_type}")
        self.logger.info(f"Kế hoạch thực thi: {len(execution_plan)} bước")
        
        return task_type, execution_plan
    
    def _simple_task_classification(self, query: str) -> str:
        """Phân loại tác vụ đơn giản dựa trên từ khóa."""
        query = query.lower()
        
        if any(kw in query for kw in ["polyp", "tổn thương", "khối", "u", "phát hiện"]):
            return "polyp_detection"
        
        if any(kw in query for kw in ["bli", "wli", "fice", "lci", "kỹ thuật", "phương pháp nội soi"]):
            return "modality_classification"
        
        if any(kw in query for kw in ["vị trí", "hang vị", "thân vị", "tâm vị", "bờ cong", "thực quản"]):
            return "region_classification"
        
        if any(kw in query for kw in ["?", "tại sao", "làm sao", "như thế nào", "mức độ", "chẩn đoán"]):
            return "medical_qa"
        
        # Mặc định là comprehensive
        return "comprehensive"
    
    def _create_execution_plan(self, task_type: str, query: Optional[str] = None) -> List[Dict]:
        """Tạo kế hoạch thực thi dựa trên loại tác vụ."""
        # Kế hoạch mặc định cho mỗi loại tác vụ
        plans = {
            "polyp_detection": [
                {"agent": "detector", "params": {"confidence_threshold": 0.3}}
            ],
            
            "modality_classification": [
                {"agent": "classifier_2", "params": {}}
            ],
            
            "region_classification": [
                {"agent": "classifier_1", "params": {}}
            ],
            
            "medical_qa": [
                {"agent": "detector", "params": {"confidence_threshold": 0.3}},
                {"agent": "vqa", "params": {"query": query}}
            ],
            
            "comprehensive": [
                {"agent": "detector", "params": {}},
                {"agent": "classifier_1", "params": {}},
                {"agent": "classifier_2", "params": {}},
                {"agent": "vqa", "params": {"query": query}} if query else {"agent": "rag", "params": {}}
            ]
        }
        
        return plans.get(task_type, plans["comprehensive"])
    
    def _execute_perception_agents_parallel(self, image_path: str) -> Dict[str, Any]:
        """Thực thi song song các agents perception."""
        import concurrent.futures
        
        results = {}
        perception_agents = ["detector", "classifier_1", "classifier_2"]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Tạo dictionary của các futures
            future_to_agent = {
                executor.submit(self._execute_single_agent, agent_name, image_path, {}): agent_name
                for agent_name in perception_agents if agent_name in self.agents
            }
            
            # Xử lý kết quả khi hoàn tất
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    agent_result = future.result()
                    results[agent_name] = agent_result
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} gặp lỗi: {str(e)}")
                    results[agent_name] = {"error": str(e), "success": False}
        
        return results
    
    def _execute_single_agent(self, agent_name: str, image_path: str, params: Dict) -> Dict[str, Any]:
        """Thực thi một agent đơn lẻ."""
        agent = self.agents.get(agent_name)
        if not agent:
            self.logger.warning(f"Agent {agent_name} không tồn tại")
            return {"error": f"Agent {agent_name} không tồn tại", "success": False}
        
        # Chuẩn bị input cho agent
        input_data = {"image_path": image_path, **params}
        
        # Gọi agent
        self.logger.info(f"Thực thi agent {agent_name}")
        return agent.process(input_data)
    
    def _execute_step(self, step: Dict, image_path: str, query: Optional[str], 
                     medical_context: Optional[Dict], previous_results: Dict) -> Dict[str, Any]:
        """
        Thực thi một bước trong kế hoạch.
        
        Args:
            step: Bước cần thực thi
            image_path: Đường dẫn hình ảnh
            query: Câu hỏi
            medical_context: Thông tin y tế
            previous_results: Kết quả từ các bước trước
            
        Returns:
            Kết quả của bước
        """
        agent_name = step["agent"]
        params = step["params"].copy()
        
        # Chuẩn bị context từ kết quả trước đó
        context = {}
        
        # Thêm kết quả detector vào context nếu có
        if "detector" in previous_results:
            if "objects" in previous_results["detector"]:
                context["detections"] = previous_results["detector"]["objects"]
            elif "detections" in previous_results["detector"]:
                context["detections"] = previous_results["detector"]["detections"]
        
        # Thêm kết quả classifier vào context nếu có
        if "classifier_1" in previous_results:
            context["region"] = previous_results["classifier_1"].get("class", "")
            context["region_confidence"] = previous_results["classifier_1"].get("confidence", 0)
        
        if "classifier_2" in previous_results:
            context["modality"] = previous_results["classifier_2"].get("class", "")
            context["modality_confidence"] = previous_results["classifier_2"].get("confidence", 0)
        
        # Thêm medical_context nếu có
        if medical_context:
            context["medical_context"] = medical_context
        
        # Xử lý đặc biệt cho VQA agent
        if agent_name == "vqa" and query:
            params["query"] = query
            
            # Nếu đã có kết quả reflection, không cần gọi lại VQA
            if "reflection" in previous_results:
                return {"vqa": previous_results.get("reflection", {})}
        
        # Chuẩn bị input cho agent
        input_data = {"image_path": image_path, **params, "context": context}
        
        # Gọi agent
        agent = self.agents.get(agent_name)
        if not agent:
            self.logger.warning(f"Agent {agent_name} không tồn tại")
            return {agent_name: {"error": f"Agent {agent_name} không tồn tại", "success": False}}
        
        self.logger.info(f"Thực thi agent {agent_name}")
        result = agent.process(input_data)
        
        return {agent_name: result}
    
    def _needs_reflection(self, query: str, vqa_result: Dict[str, Any]) -> bool:
        """
        Kiểm tra xem kết quả từ VQA có cần reflection không.
        
        Args:
            query: Câu hỏi
            vqa_result: Kết quả từ VQA agent
        Returns:
            bool: True nếu cần reflection
        """
        # Kiểm tra 1: Độ tin cậy thấp
        if "confidence" in vqa_result and vqa_result["confidence"] < 0.7:
            self.logger.info(f"Phát hiện độ tin cậy thấp: {vqa_result['confidence']}")
            return True
        
        # Kiểm tra 2: Biểu hiện không chắc chắn trong câu trả lời
        if "answer" in vqa_result:
            answer = vqa_result["answer"].lower()
            uncertainty_phrases = ["có thể", "không chắc chắn", "khó xác định", "có lẽ"]
            
            if any(phrase in answer for phrase in uncertainty_phrases):
                self.logger.info("Phát hiện biểu hiện không chắc chắn trong câu trả lời")
                return True
        
        # Kiểm tra 3: Tìm câu hỏi tương tự trong lịch sử
        if self.config.memory_enabled:
            similar_questions = self._find_similar_questions(query)
            
            if similar_questions:
                self.logger.info(f"Tìm thấy {len(similar_questions)} câu hỏi tương tự")
                return True
        
        return False
    
    def _find_similar_questions(self, query: str) -> List[Dict[str, Any]]:
        """
        Tìm các câu hỏi tương tự trong lịch sử.
        
        Args:
            query: Câu hỏi hiện tại
            
        Returns:
            List các câu hỏi tương tự với câu trả lời
        """
        if not self.config.memory_enabled or not hasattr(self, 'vector_store'):
            return []
        
        try:
            similar_items = self.vector_store.search(
                query, 
                limit=5, 
                threshold=self.config.similarity_threshold
            )
            
            # Lọc các items không phải là câu hỏi
            similar_questions = []
            
            for item in similar_items:
                if item.get("type") == "question" and item.get("text") != query:
                    # Lấy câu trả lời tương ứng
                    answer = self.conversation_memory.get_answer(item.get("id"))
                    
                    if answer:
                        similar_questions.append({
                            "question": item.get("text"),
                            "answer": answer,
                            "similarity": item.get("score", 0)
                        })
            
            return similar_questions
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tìm câu hỏi tương tự: {str(e)}")
            return []
    
    def _perform_reflection(self, query: str, vqa_result: Dict[str, Any], 
                          medical_context: Optional[Dict], other_results: Dict) -> Dict[str, Any]:
        """
        Thực hiện reflection để cải thiện kết quả từ VQA.
        
        Args:
            query: Câu hỏi
            vqa_result: Kết quả từ VQA agent
            medical_context: Thông tin y tế bổ sung
            other_results: Kết quả từ các agents khác
            
        Returns:
            Dict: Kết quả sau khi reflection
        """
        # Nếu không có LLM client, không thể thực hiện reflection
        if not self.llm_client:
            return vqa_result
        
        try:
            # Tạo context cho reflection
            reflection_context = {}
            
            # Thêm kết quả VQA
            if "answer" in vqa_result:
                reflection_context["original_answer"] = vqa_result["answer"]
            
            # Tìm các câu hỏi tương tự
            similar_questions = self._find_similar_questions(query)
            
            # Tạo prompt cho reflection
            prompt = self._create_reflection_prompt(
                query, vqa_result, similar_questions, medical_context, other_results
            )
            
            # Gọi LLM để thực hiện reflection
            reflection_response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            reflection_analysis = reflection_response.choices[0].message.content
            
            # Tạo prompt cho câu trả lời cải thiện
            improved_prompt = f"""
            Dựa trên phân tích trên, hãy đưa ra câu trả lời y tế cải thiện, chính xác và nhất quán nhất cho câu hỏi:
            
            "{query}"
            
            Câu trả lời phải dựa trên bằng chứng y tế, rõ ràng và tránh các biểu hiện không chắc chắn trừ khi thực sự cần thiết.
            
            Trả lời:
            """
            
            improved_response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reflection_analysis},
                    {"role": "user", "content": improved_prompt}
                ],
                temperature=0.2
            )
            
            improved_answer = improved_response.choices[0].message.content
            
            # Kết quả reflection
            reflection_result = {
                "original_answer": vqa_result.get("answer", ""),
                "improved_answer": improved_answer,
                "analysis": reflection_analysis,
                "bias_detected": len(similar_questions) > 0,
                "similar_questions": similar_questions if similar_questions else [],
                "confidence": 0.9  # Độ tin cậy cao hơn sau khi reflection
            }
            
            return reflection_result
            
        except Exception as e:
            self.logger.error(f"Lỗi trong quá trình reflection: {str(e)}")
            return vqa_result
    
    def _create_reflection_prompt(self, query: str, vqa_result: Dict[str, Any], 
                                similar_questions: List[Dict], medical_context: Optional[Dict],
                                other_results: Dict) -> str:
        """Tạo prompt cho reflection."""
        prompt = f"""
        Bạn là chuyên gia y tế cao cấp, đang phân tích câu trả lời của AI cho một câu hỏi về hình ảnh y tế.
        
        Câu hỏi hiện tại: "{query}"
        
        Câu trả lời hiện tại: "{vqa_result.get('answer', '')}"
        """
        
        # Thêm thông tin về độ tin cậy
        if "confidence" in vqa_result:
            prompt += f"\nĐộ tin cậy: {vqa_result['confidence']:.2f}"
        
        # Thêm các câu hỏi tương tự nếu có
        if similar_questions:
            prompt += "\n\nCác câu hỏi tương tự và câu trả lời tương ứng:"
            
            for idx, item in enumerate(similar_questions, 1):
                prompt += f"\n\nCâu hỏi tương tự {idx}: \"{item['question']}\""
                prompt += f"\nĐộ tương đồng: {item['similarity']:.2f}"
                prompt += f"\nCâu trả lời: \"{item['answer']}\""
        
        # Thêm context từ detector
        if "detector" in other_results and other_results["detector"].get("objects"):
            prompt += "\n\nKết quả phát hiện đối tượng:"
            
            for idx, obj in enumerate(other_results["detector"]["objects"][:3], 1):
                prompt += f"\n- Đối tượng {idx}: {obj.get('class', 'không xác định')} (độ tin cậy: {obj.get('confidence', 0):.2f})"
                if "position_description" in obj:
                    prompt += f", vị trí: {obj['position_description']}"
        
        # Thêm context từ classifiers
        if "classifier_1" in other_results:
            cls1 = other_results["classifier_1"]
            prompt += f"\n\nVị trí trong đường tiêu hóa: {cls1.get('class', 'không xác định')} (độ tin cậy: {cls1.get('confidence', 0):.2f})"
        
        if "classifier_2" in other_results:
            cls2 = other_results["classifier_2"]
            prompt += f"\n\nKỹ thuật nội soi: {cls2.get('class', 'không xác định')} (độ tin cậy: {cls2.get('confidence', 0):.2f})"
        
        # Thêm medical context nếu có
        if medical_context:
            prompt += "\n\nThông tin y tế bổ sung:"
            for key, value in medical_context.items():
                prompt += f"\n- {key}: {value}"
        
        # Hướng dẫn reflection
        prompt += """
        
        Nhiệm vụ của bạn là:
        
        1. Phân tích chất lượng và độ tin cậy của câu trả lời
        2. Nếu có các câu hỏi tương tự, so sánh các câu trả lời và phát hiện mâu thuẫn hoặc bias
        3. Đánh giá câu trả lời dựa trên kết quả từ các agents khác (detector, classifiers)
        4. Chỉ ra những điểm không chắc chắn và cần cải thiện
        
        Hãy phân tích một cách chi tiết, khách quan và dựa trên bằng chứng y tế. Nhận xét về tính nhất quán, chính xác và đầy đủ của câu trả lời.
        """
        
        return prompt
    
    def _synthesize_results(self, task_type: str, results: Dict[str, Any], query: Optional[str]) -> Dict[str, Any]:
        """
        Tổng hợp kết quả từ các agents thành kết quả cuối cùng.
        
        Args:
            task_type: Loại tác vụ
            results: Kết quả từ các agents
            query: Câu hỏi ban đầu
            
        Returns:
            Dict: Kết quả cuối cùng đã tổng hợp
        """
        final_result = {
            "task_type": task_type,
            "success": True
        }
        
        # Thêm kết quả từ detector nếu có
        if "detector" in results:
            detector_result = results["detector"]
            
            if "objects" in detector_result:
                final_result["polyps"] = detector_result["objects"]
                final_result["polyp_count"] = len(detector_result["objects"])
            elif "detections" in detector_result:
                final_result["polyps"] = detector_result["detections"]
                final_result["polyp_count"] = len(detector_result["detections"])
            else:
                final_result["polyps"] = []
                final_result["polyp_count"] = 0
        
        # Thêm kết quả từ classifiers nếu có
        if "classifier_1" in results:
            cls1 = results["classifier_1"]
            final_result["region"] = {
                "class": cls1.get("class", "unknown"),
                "confidence": cls1.get("confidence", 0),
                "uncertain": cls1.get("uncertain", False)
            }
        
        if "classifier_2" in results:
            cls2 = results["classifier_2"]
            final_result["modality"] = {
                "class": cls2.get("class", "unknown"),
                "confidence": cls2.get("confidence", 0),
                "uncertain": cls2.get("uncertain", False)
            }
        
        # Thêm kết quả từ VQA hoặc Reflection
        if "reflection" in results:
            reflection = results["reflection"]
            final_result["answer"] = reflection.get("improved_answer", "")
            final_result["analysis"] = reflection.get("analysis", "")
            final_result["original_answer"] = reflection.get("original_answer", "")
            final_result["bias_detected"] = reflection.get("bias_detected", False)
        elif "vqa" in results:
            vqa = results["vqa"]
            final_result["answer"] = vqa.get("answer", "")
            final_result["confidence"] = vqa.get("confidence", 0)
        
        # Tạo tóm tắt nếu là tác vụ comprehensive
        if task_type == "comprehensive":
            final_result["summary"] = self._generate_summary(final_result, query)
        
        return final_result
    
    def _generate_summary(self, result: Dict[str, Any], query: Optional[str]) -> str:
        """Tạo bản tóm tắt từ kết quả phân tích."""
        summary = []
        
        # Tóm tắt về polyps
        if "polyps" in result and result["polyps"]:
            polyp_count = result["polyp_count"]
            summary.append(f"Phát hiện {polyp_count} polyp trong hình ảnh.")
            
            # Mô tả polyp lớn nhất
            largest_polyp = max(result["polyps"], key=lambda p: p.get("area", 0))
            confidence = largest_polyp.get("confidence", 0)
            position = largest_polyp.get("position_description", "không xác định")
            
            summary.append(f"Polyp lớn nhất nằm ở vị trí {position} với độ tin cậy {confidence:.2f}.")
        else:
            summary.append("Không phát hiện polyp nào trong hình ảnh.")
        
        # Tóm tắt về region
        if "region" in result:
            region = result["region"]
            summary.append(f"Hình ảnh được chụp tại vị trí {region['class']} với độ tin cậy {region['confidence']:.2f}.")
        
        # Tóm tắt về modality
        if "modality" in result:
            modality = result["modality"]
            summary.append(f"Kỹ thuật nội soi sử dụng là {modality['class']} với độ tin cậy {modality['confidence']:.2f}.")
        
        # Tóm tắt câu trả lời nếu có
        if query and "answer" in result:
            summary.append(f"Trả lời cho câu hỏi: {result['answer']}")
        
        # Cảnh báo về bias nếu có
        if result.get("bias_detected", False):
            summary.append("Lưu ý: Đã phát hiện và điều chỉnh bias trong câu trả lời.")
        
        return "\n".join(summary)
    
    def _store_in_memory(self, image_path: str, query: Optional[str], 
                       results: Dict[str, Any], final_result: Dict[str, Any],
                       medical_context: Optional[Dict]) -> None:
        """Lưu kết quả vào memory."""
        if not self.config.memory_enabled:
            return
        
        try:
            # Lưu vào conversation memory
            session_data = {
                "id": self.current_session_id,
                "timestamp": time.time(),
                "image_path": image_path,
                "query": query,
                "results": results,
                "final_result": final_result,
                "medical_context": medical_context
            }
            
            self.conversation_memory.add_session(session_data)
            
            # Lưu vào vector store nếu có query
            if query:
                # Lưu query
                self.vector_store.add_item(
                    text=query,
                    metadata={
                        "id": self.current_session_id,
                        "type": "question",
                        "timestamp": time.time()
                    }
                )
                
                # Lưu answer
                answer = final_result.get("answer", "")
                if answer:
                    self.vector_store.add_item(
                        text=answer,
                        metadata={
                            "id": self.current_session_id,
                            "type": "answer",
                            "timestamp": time.time(),
                            "question_id": self.current_session_id
                        }
                    )
            
            self.logger.info(f"Đã lưu session {self.current_session_id} vào memory")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu vào memory: {str(e)}")


# File này có thể được sử dụng như một module hoặc script độc lập
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Orchestrator")
    parser.add_argument("--image", required=True, help="Đường dẫn đến hình ảnh nội soi")
    parser.add_argument("--query", help="Câu hỏi y tế (tùy chọn)")
    parser.add_argument("--output", default="results", help="Thư mục đầu ra")
    parser.add_argument("--api_key", help="OpenAI API key cho reflection")
    
    args = parser.parse_args()
    
    # Khởi tạo orchestrator
    config = OrchestratorConfig(
        name="Medical Orchestrator",
        output_path=args.output,
        llm_api_key=args.api_key
    )
    
    orchestrator = MedicalOrchestrator(config)
    
    # Khởi tạo và đăng ký các agents
    from agents.detector import DetectorAgent, DetectorAgentConfig
    detector_config = DetectorAgentConfig(
        name="Polyp Detector",
        model_path="models/yolov8n.pt",  # Thay đổi đường dẫn model nếu cần
        confidence_threshold=0.3
    )
    detector = DetectorAgent(detector_config)
    detector.initialize()
    orchestrator.register_agent("detector", detector)
    
    from agents.classifier_1 import MedicalClassifierAgent1, ClassifierConfig1
    classifier1_config = ClassifierConfig1(
        model_path="models/region_classifier.pt"  # Thay đổi đường dẫn model nếu cần
    )
    classifier1 = MedicalClassifierAgent1(classifier1_config)
    orchestrator.register_agent("classifier_1", classifier1)
    
    from agents.classifier_2 import MedicalClassifierAgent2, ClassifierConfig2
    classifier2_config = ClassifierConfig2(
        model_path="models/modality_classifier.pt"  # Thay đổi đường dẫn model nếu cần
    )
    classifier2 = MedicalClassifierAgent2(classifier2_config)
    orchestrator.register_agent("classifier_2", classifier2)
    
    from agents.vqa import MedicalVQAAgent, VQAAgentConfig
    vqa_config = VQAAgentConfig(
        name="Medical VQA",
        model_path="models/llava_med.pt"  # Thay đổi đường dẫn model nếu cần
    )
    vqa = MedicalVQAAgent(vqa_config)
    vqa.initialize()
    orchestrator.register_agent("vqa", vqa)
    
    # Thực thi orchestrator
    result = orchestrator.orchestrate(args.image, args.query)
    
    # In kết quả
    print(json.dumps(result, indent=2, ensure_ascii=False))