#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reflexion Planner Agent  
-----------------------
Agent chịu trách nhiệm lập kế hoạch thực thi cho hệ thống Reflexion.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import openai

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator.reflexion.config import (
    ExecutionPlan, ExecutionStep, AgentType, StepStatus, ReflexionConfig
)

class PlannerAgent:
    """
    Planner Agent sử dụng LLM để tạo kế hoạch thực thi optimal.
    
    Nhiệm vụ:
    1. Phân tích query và image path
    2. Quyết định agents nào cần gọi
    3. Xác định thứ tự thực thi và dependencies
    4. Ước lượng thời gian và resources cần thiết
    """
    
    def __init__(self, config: ReflexionConfig):
        """Khởi tạo Planner Agent."""
        self.config = config
        self.logger = logging.getLogger(f"planner.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.logging_level))
        
        # Khởi tạo OpenAI client
        self.llm_client = self._initialize_llm()
        
        # Template prompts
        self.system_prompt = self._create_system_prompt()
        self.planning_prompt_template = self._create_planning_prompt_template()
        
        # Metrics
        self.metrics = {
            "total_plans_created": 0,
            "successful_plans": 0,
            "average_planning_time": 0.0,
            "api_calls": 0
        }
        
        self.logger.info("Planner Agent initialized successfully")
    
    def _initialize_llm(self) -> openai.OpenAI:
        """Khởi tạo LLM client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return openai.OpenAI(api_key=api_key)
    
    def _create_system_prompt(self) -> str:
        """Tạo system prompt cho Planner."""
        return """Bạn là Planning Expert trong hệ thống Medical AI multi-agent chuyên phân tích hình ảnh nội soi.

Nhiệm vụ của bạn:
1. Phân tích query của người dùng về hình ảnh y tế
2. Lập kế hoạch thực thi tối ưu với các agents có sẵn
3. Xác định thứ tự gọi agents và dependencies

Các agents có sẵn:
- **detector**: Phát hiện polyp và objects trong hình ảnh nội soi (YOLO-based)
- **classifier_1**: Phân loại kỹ thuật nội soi (WLI, BLI, FICE, LCI)  
- **classifier_2**: Phân loại vị trí giải phẫu (thực quản, dạ dày, tá tràng, đại tràng)
- **vqa**: Trả lời câu hỏi về hình ảnh (LLaVA-based)
- **rag**: Trả lời câu hỏi dựa trên knowledge base y tế

Nguyên tắc lập kế hoạch:
1. Luôn bắt đầu với detector nếu query liên quan đến phát hiện objects
2. Classifiers chạy song song sau detector
3. VQA chạy sau khi có context từ detection và classification
4. RAG dùng khi cần thông tin y tế chuyên sâu
5. Tối ưu hóa thời gian bằng cách chạy parallel khi có thể

Trả lời theo format JSON:
{
    "reasoning": "Giải thích lý do thiết kế plan này",
    "estimated_time": "Thời gian ước tính (giây)",
    "priority_level": "Mức độ ưu tiên (1-10)",
    "steps": [
        {
            "agent_type": "detector",
            "params": {"confidence_threshold": 0.5},
            "dependencies": [],
            "estimated_time": 5.0
        }
    ]
}"""
    
    def _create_planning_prompt_template(self) -> str:
        """Tạo template cho planning prompt."""
        return """**MEDICAL AI PLANNING REQUEST**

**Query**: {query}
**Image Path**: {image_path}
**Previous Iteration**: {iteration_count}
**Previous Issues**: {previous_issues}

**Context từ iteration trước**:
{previous_context}

**Yêu cầu**: Hãy tạo execution plan tối ưu cho query này.

Lưu ý:
- Nếu đây là iteration đầu tiên, tập trung vào phân tích toàn diện
- Nếu là retry, tập trung vào những điểm cần cải thiện từ feedback
- Đảm bảo độ chính xác cao cho medical domain
- Tối ưu thời gian thực thi"""
    
    def create_plan(self, 
                   query: str, 
                   image_path: str, 
                   iteration_count: int = 0,
                   previous_feedback: Optional[List[str]] = None,
                   previous_results: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Tạo kế hoạch thực thi dựa trên query và context.
        
        Args:
            query: Câu hỏi/yêu cầu từ người dùng
            image_path: Đường dẫn đến hình ảnh
            iteration_count: Số lần iteration (cho refinement)
            previous_feedback: Feedback từ critic ở iteration trước
            previous_results: Kết quả từ iteration trước
            
        Returns:
            ExecutionPlan: Kế hoạch thực thi chi tiết
        """
        start_time = time.time()
        
        try:
            # Chuẩn bị context cho LLM
            previous_issues = self._extract_issues(previous_feedback) if previous_feedback else "Không có"
            previous_context = self._format_previous_context(previous_results) if previous_results else "Lần đầu thực thi"
            
            # Tạo prompt
            prompt = self.planning_prompt_template.format(
                query=query,
                image_path=image_path,
                iteration_count=iteration_count,
                previous_issues=previous_issues,
                previous_context=previous_context
            )
            
            # Gọi LLM
            response = self._call_llm(prompt)
            
            # Parse response thành ExecutionPlan
            plan = self._parse_llm_response(response, query, image_path)
            plan.version = iteration_count + 1
            
            # Cập nhật metrics
            planning_time = time.time() - start_time
            self._update_metrics(planning_time, success=True)
            
            self.logger.info(f"Created plan with {len(plan.steps)} steps (v{plan.version})")
            self.logger.debug(f"Plan reasoning: {plan.reasoning}")
            
            return plan
            
        except Exception as e:
            planning_time = time.time() - start_time
            self._update_metrics(planning_time, success=False)
            
            self.logger.error(f"Failed to create plan: {str(e)}")
            
            # Fallback plan
            return self._create_fallback_plan(query, image_path)
    
    def _call_llm(self, prompt: str) -> str:
        """Gọi LLM để tạo plan."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.planner_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.planner_temperature,
                max_tokens=2000
            )
            
            self.metrics["api_calls"] += 1
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str, query: str, image_path: str) -> ExecutionPlan:
        """Parse response từ LLM thành ExecutionPlan."""
        try:
            # Hỗ trợ multiple JSON formats
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end] if json_start != -1 else response
            
            data = json.loads(json_str)
            
            # Tạo ExecutionPlan
            plan = ExecutionPlan(
                query=query,
                image_path=image_path,
                reasoning=data.get("reasoning", ""),
                estimated_time=float(data.get("estimated_time", 60.0)),
                priority_level=int(data.get("priority_level", 5))
            )
            
            # Tạo ExecutionSteps
            for step_data in data.get("steps", []):
                step = ExecutionStep(
                    agent_type=AgentType(step_data["agent_type"]),
                    params=step_data.get("params", {}),
                    dependencies=step_data.get("dependencies", []),
                    timeout=float(step_data.get("estimated_time", self.config.step_timeout)),
                    max_retries=step_data.get("max_retries", 3)
                )
                plan.steps.append(step)
            
            return plan
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            return self._create_fallback_plan(query, image_path)
    
    def _create_fallback_plan(self, query: str, image_path: str) -> ExecutionPlan:
        """Tạo fallback plan khi LLM fail."""
        plan = ExecutionPlan(
            query=query,
            image_path=image_path,
            reasoning="Fallback plan - LLM unavailable",
            estimated_time=120.0,
            priority_level=5
        )
        
        # Plan mặc định: comprehensive analysis
        steps = [
            ExecutionStep(agent_type=AgentType.DETECTOR, params={}),
            ExecutionStep(agent_type=AgentType.CLASSIFIER_1, params={}, dependencies=["detector"]),
            ExecutionStep(agent_type=AgentType.CLASSIFIER_2, params={}, dependencies=["detector"])
        ]
        
        # Thêm VQA nếu có query
        if query and len(query.strip()) > 0:
            vqa_step = ExecutionStep(
                agent_type=AgentType.VQA, 
                params={"query": query},
                dependencies=["detector", "classifier_1", "classifier_2"]
            )
            steps.append(vqa_step)
        
        plan.steps = steps
        return plan
    
    def _extract_issues(self, feedback: List[str]) -> str:
        """Trích xuất issues từ feedback."""
        if not feedback:
            return "Không có"
        
        # Lọc và format issues
        issues = [f"- {issue}" for issue in feedback if issue.strip()]
        return "\n".join(issues) if issues else "Không có"
    
    def _format_previous_context(self, results: Dict[str, Any]) -> str:
        """Format kết quả từ iteration trước."""
        if not results:
            return "Không có kết quả từ iteration trước"
        
        context = []
        
        # Format detection results
        if "detector" in results:
            detector_result = results["detector"]
            if detector_result.get("success") and detector_result.get("objects"):
                polyp_count = len(detector_result["objects"])
                context.append(f"- Detector: Phát hiện {polyp_count} polyp")
            else:
                context.append("- Detector: Không phát hiện polyp")
        
        # Format classification results
        if "classifier_1" in results:
            cls1 = results["classifier_1"]
            context.append(f"- Kỹ thuật nội soi: {cls1.get('class', 'Unknown')} ({cls1.get('confidence', 0):.2f})")
        
        if "classifier_2" in results:
            cls2 = results["classifier_2"]
            context.append(f"- Vị trí: {cls2.get('class', 'Unknown')} ({cls2.get('confidence', 0):.2f})")
        
        # Format VQA results
        if "vqa" in results:
            vqa = results["vqa"]
            context.append(f"- VQA: {vqa.get('answer', 'Không có câu trả lời')[:100]}...")
        
        return "\n".join(context) if context else "Không có thông tin"
    
    def _update_metrics(self, planning_time: float, success: bool):
        """Cập nhật metrics."""
        self.metrics["total_plans_created"] += 1
        if success:
            self.metrics["successful_plans"] += 1
        
        # Cập nhật average time
        total_time = self.metrics["average_planning_time"] * (self.metrics["total_plans_created"] - 1)
        self.metrics["average_planning_time"] = (total_time + planning_time) / self.metrics["total_plans_created"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Lấy metrics của Planner."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_plans"] / max(1, self.metrics["total_plans_created"])
            ) * 100
        }
    
    def optimize_plan(self, plan: ExecutionPlan, performance_data: Dict[str, Any]) -> ExecutionPlan:
        """
        Tối ưu hóa plan dựa trên performance data từ các lần chạy trước.
        
        Args:
            plan: Plan cần tối ưu
            performance_data: Dữ liệu hiệu suất từ các lần thực thi trước
            
        Returns:
            ExecutionPlan: Plan đã được tối ưu
        """
        # Điều chỉnh timeout dựa trên historical data
        for step in plan.steps:
            agent_type = step.agent_type.value
            if agent_type in performance_data:
                avg_time = performance_data[agent_type].get("average_time", 30.0)
                step.timeout = max(step.timeout, avg_time * 1.5)  # Buffer 50%
        
        # Điều chỉnh confidence thresholds
        if "detector_performance" in performance_data:
            detector_steps = [s for s in plan.steps if s.agent_type == AgentType.DETECTOR]
            for step in detector_steps:
                # Tăng confidence threshold nếu có nhiều false positives
                false_positive_rate = performance_data["detector_performance"].get("false_positive_rate", 0.1)
                if false_positive_rate > 0.2:
                    current_threshold = step.params.get("confidence_threshold", 0.5)
                    step.params["confidence_threshold"] = min(0.9, current_threshold + 0.1)
        
        return plan