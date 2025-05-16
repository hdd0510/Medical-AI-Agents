#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reflexion Critic Agent
----------------------
Agent chịu trách nhiệm đánh giá kết quả và đưa ra feedback để cải thiện.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import openai
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator.reflexion.types import (
    ExecutionPlan, ExecutionStep, StepResult, CriticFeedback, 
    CriticScore, AgentType, ReflexionConfig
)

class CriticAgent:
    """
    Critic Agent sử dụng LLM để đánh giá chất lượng kết quả từ các agents.
    
    Nhiệm vụ:
    1. Đánh giá tổng thể kết quả execution
    2. Phát hiện inconsistencies và errors
    3. So sánh với domain knowledge y tế
    4. Đề xuất cải thiện cụ thể
    5. Quyết định có cần iteration tiếp theo không
    """
    
    def __init__(self, config: ReflexionConfig):
        """Khởi tạo Critic Agent."""
        self.config = config
        self.logger = logging.getLogger(f"critic.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.logging_level))
        
        # Khởi tạo OpenAI client
        self.llm_client = self._initialize_llm()
        
        # Template prompts
        self.system_prompt = self._create_system_prompt()
        self.criticism_prompt_template = self._create_criticism_prompt_template()
        
        # Medical domain knowledge cơ bản
        self.medical_rules = self._load_medical_rules()
        
        # Metrics
        self.metrics = {
            "total_evaluations": 0,
            "high_quality_results": 0,
            "identified_issues": 0,  
            "suggested_improvements": 0,
            "api_calls": 0,
            "average_evaluation_time": 0.0
        }
        
        self.logger.info("Critic Agent initialized successfully")
    
    def _initialize_llm(self) -> openai.OpenAI:
        """Khởi tạo LLM client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return openai.OpenAI(api_key=api_key)
    
    def _create_system_prompt(self) -> str:
        """Tạo system prompt cho Critic."""
        return """Bạn là Medical AI Critic chuyên đánh giá kết quả phân tích hình ảnh nội soi.

Vai trò của bạn:
1. Đánh giá chất lượng kết quả từ multi-agent system
2. Phát hiện lỗi logic, inconsistencies, hoặc medical inaccuracies
3. So sánh với medical domain knowledge
4. Đề xuất cải thiện cụ thể

Tiêu chí đánh giá:
- **Accuracy**: Kết quả có chính xác không?
- **Consistency**: Các agents có nhất quán không?
- **Completeness**: Có hoàn thành yêu cầu không?
- **Clinical Relevance**: Có giá trị lâm sàng không?
- **Safety**: Có safe cho medical decision making không?

Scoring system:
- EXCELLENT (5): Hoàn hảo, không cần cải thiện
- GOOD (4): Tốt, có thể improve nhỏ
- ACCEPTABLE (3): Chấp nhận được
- POOR (2): Nhiều vấn đề, cần cải thiện 
- FAILURE (1): Thất bại, cần làm lại

Trả lời theo format JSON:
{
    "overall_score": "ACCEPTABLE",
    "step_scores": {
        "detector": "GOOD",
        "classifier_1": "EXCELLENT"
    },
    "issues": [
        "Detection confidence quá thấp (0.3)",
        "Inconsistency giữa classifier outputs"
    ],
    "suggestions": [
        "Tăng confidence threshold của detector lên 0.5",
        "Chạy lại classifier với feature refinement"
    ],
    "reasoning": "Giải thích chi tiết đánh giá",
    "requires_refinement": true,
    "refinement_priority": [
        {"step": "detector", "reason": "Low confidence detections"},
        {"step": "classifier_1", "reason": "Inconsistent classification"}
    ]
}"""
    
    def _create_criticism_prompt_template(self) -> str:
        """Tạo template cho criticism prompt."""
        return """**MEDICAL AI EVALUATION REQUEST**

**Original Query**: {query}
**Image Path**: {image_path}
**Execution Plan**: {plan_summary}
**Iteration**: {iteration_count}

**RESULTS TO EVALUATE**:
{results_summary}

**CROSS-VALIDATION**:
{cross_validation}

**MEDICAL CONTEXT**:
- Target organ: GI tract endoscopy
- Clinical significance: Polyp detection and classification
- Safety requirements: High precision, minimal false negatives

**Tasks:**
1. Evaluate each step's results against medical standards
2. Check consistency between agents (e.g., VQA answer matches detection results)  
3. Identify any safety concerns or medical inaccuracies
4. Suggest specific improvements with rationale

**Previous Issues** (if any): {previous_issues}

Please provide a comprehensive evaluation following the JSON format."""
    
    def _load_medical_rules(self) -> Dict[str, Any]:
        """Load basic medical domain rules."""
        return {
            "polyp_confidence_threshold": 0.7,  # Ngưỡng confidence tối thiểu cho polyp detection
            "classification_consistency": 0.8,  # Ngưỡng consistency giữa các classifiers
            "vqa_medical_keywords": [
                "polyp", "adenoma", "hyperplastic", "sessile", "pedunculated",
                "carcinoma", "dysplasia", "normal tissue", "inflammatory"
            ],
            "anatomical_locations": [
                "esophagus", "stomach", "duodenum", "colon", "rectum",
                "antrum", "body", "fundus", "cardia"
            ],
            "imaging_modalities": ["WLI", "BLI", "NBI", "FICE", "LCI"],
            "size_thresholds": {
                "small": 5,  # mm
                "medium": 10,
                "large": 20
            }
        }
    
    def evaluate_results(self, 
                        plan: ExecutionPlan,
                        results: List[StepResult],
                        iteration_count: int = 0,
                        previous_feedback: Optional[List[CriticFeedback]] = None) -> CriticFeedback:
        """
        Đánh giá kết quả execution và đưa ra feedback.
        
        Args:
            plan: Execution plan đã thực thi
            results: Kết quả từ các steps
            iteration_count: Số iteration hiện tại
            previous_feedback: Feedback từ iterations trước
            
        Returns:
            CriticFeedback: Đánh giá chi tiết và suggestions
        """
        start_time = time.time()
        
        try:
            # Chuẩn bị context cho LLM
            plan_summary = self._summarize_plan(plan)
            results_summary = self._summarize_results(results)
            cross_validation = self._perform_cross_validation(results)
            previous_issues = self._extract_previous_issues(previous_feedback)
            
            # Tạo prompt
            prompt = self.criticism_prompt_template.format(
                query=plan.query,
                image_path=plan.image_path,
                plan_summary=plan_summary,
                iteration_count=iteration_count,
                results_summary=results_summary,
                cross_validation=cross_validation,
                previous_issues=previous_issues
            )
            
            # Gọi LLM
            response = self._call_llm(prompt)
            
            # Parse response thành CriticFeedback
            feedback = self._parse_llm_response(response, plan.id)
            
            # Validation và enhancement
            feedback = self._enhance_feedback(feedback, plan, results)
            
            # Cập nhật metrics
            evaluation_time = time.time() - start_time
            self._update_metrics(evaluation_time, feedback.overall_score)
            
            self.logger.info(f"Evaluated results: {feedback.overall_score.name} score")
            self.logger.info(f"Found {len(feedback.issues)} issues, {len(feedback.suggestions)} suggestions")
            
            return feedback
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            self._update_metrics(evaluation_time, CriticScore.FAILURE)
            
            self.logger.error(f"Failed to evaluate results: {str(e)}")
            
            # Fallback feedback
            return self._create_fallback_feedback(plan.id, results)
    
    def _call_llm(self, prompt: str) -> str:
        """Gọi LLM để đánh giá."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.critic_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.critic_temperature,
                max_tokens=2500
            )
            
            self.metrics["api_calls"] += 1
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str, plan_id: str) -> CriticFeedback:
        """Parse response từ LLM thành CriticFeedback."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end] if json_start != -1 else response
            
            data = json.loads(json_str)
            
            # Parse step scores
            step_scores = {}
            for step_id, score_str in data.get("step_scores", {}).items():
                try:
                    step_scores[step_id] = CriticScore[score_str.upper()]
                except KeyError:
                    self.logger.warning(f"Unknown score: {score_str}, defaulting to ACCEPTABLE")
                    step_scores[step_id] = CriticScore.ACCEPTABLE
            
            # Tạo CriticFeedback
            feedback = CriticFeedback(
                plan_id=plan_id,
                overall_score=CriticScore[data.get("overall_score", "ACCEPTABLE").upper()],
                step_scores=step_scores,
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                reasoning=data.get("reasoning", ""),
                confidence=self._calculate_confidence(data)
            )
            
            # Parse refinement plan if needed
            if data.get("requires_refinement", False):
                # Có thể tạo refinement plan từ priorities
                pass
            
            return feedback
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            return self._create_fallback_feedback(plan_id)
    
    def _summarize_plan(self, plan: ExecutionPlan) -> str:
        """Tóm tắt execution plan."""
        summary = [f"Plan v{plan.version} with {len(plan.steps)} steps:"]
        for i, step in enumerate(plan.steps):
            summary.append(f"{i+1}. {step.agent_type.value}: {step.params}")
        return "\n".join(summary)
    
    def _summarize_results(self, results: List[StepResult]) -> str:
        """Tóm tắt kết quả execution."""
        summary = []
        
        for result in results:
            agent_type = result.agent_name or "unknown"
            summary.append(f"\n**{agent_type.upper()}**:")
            summary.append(f"- Status: {result.status.value}")
            summary.append(f"- Processing time: {result.processing_time:.2f}s")
            
            if result.status.value == "success" and result.data:
                # Summarize data based on agent type
                if "detector" in agent_type.lower():
                    objects = result.data.get("objects", [])
                    summary.append(f"- Detected {len(objects)} objects")
                    for obj in objects[:3]:  # Top 3 objects
                        conf = obj.get("confidence", 0)
                        cls = obj.get("class", "unknown")
                        summary.append(f"  * {cls}: {conf:.2f}")
                
                elif "classifier" in agent_type.lower():
                    cls = result.data.get("class", "unknown")
                    conf = result.data.get("confidence", 0)
                    summary.append(f"- Classification: {cls} (confidence: {conf:.2f})")
                
                elif "vqa" in agent_type.lower():
                    answer = result.data.get("answer", "")
                    conf = result.data.get("confidence", 0)
                    summary.append(f"- Answer: {answer[:100]}...")
                    summary.append(f"- VQA confidence: {conf:.2f}")
            
            elif result.error:
                summary.append(f"- Error: {result.error}")
        
        return "\n".join(summary)
    
    def _perform_cross_validation(self, results: List[StepResult]) -> str:
        """Thực hiện cross-validation giữa các kết quả."""
        validation = []
        
        # Lấy kết quả từ các agents
        detector_result = next((r for r in results if "detect" in r.agent_name.lower()), None)
        classifier1_result = next((r for r in results if "classifier_1" in r.agent_name.lower()), None)
        classifier2_result = next((r for r in results if "classifier_2" in r.agent_name.lower()), None)
        vqa_result = next((r for r in results if "vqa" in r.agent_name.lower()), None)
        
        # Check 1: Detection - VQA consistency
        if detector_result and vqa_result and detector_result.status.value == "success":
            objects = detector_result.data.get("objects", [])
            vqa_answer = vqa_result.data.get("answer", "").lower() if vqa_result.status.value == "success" else ""
            
            has_polyp_detection = len(objects) > 0
            has_polyp_in_answer = any(keyword in vqa_answer for keyword in ["polyp", "adenoma", "lesion"])
            
            if has_polyp_detection != has_polyp_in_answer:
                validation.append("⚠️ INCONSISTENCY: Detection vs VQA mismatch")
            else:
                validation.append("✓ Detection-VQA consistency: OK")
        
        # Check 2: Classifier confidence levels
        for result in [classifier1_result, classifier2_result]:
            if result and result.status.value == "success":
                conf = result.data.get("confidence", 0)
                if conf < 0.7:
                    validation.append(f"⚠️ LOW CONFIDENCE: {result.agent_name} ({conf:.2f})")
        
        # Check 3: Medical accuracy checks
        if classifier2_result and classifier2_result.status.value == "success":
            location = classifier2_result.data.get("class", "").lower()
            if location in self.medical_rules["anatomical_locations"]:
                validation.append(f"✓ Valid anatomical location: {location}")
            elif location:
                validation.append(f"⚠️ UNKNOWN LOCATION: {location}")
        
        return "\n".join(validation) if validation else "No cross-validation performed"
    
    def _extract_previous_issues(self, previous_feedback: Optional[List[CriticFeedback]]) -> str:
        """Trích xuất issues từ feedback trước."""
        if not previous_feedback:
            return "No previous issues"
        
        all_issues = []
        for feedback in previous_feedback:
            all_issues.extend(feedback.issues)
        
        return "\n".join(f"- {issue}" for issue in all_issues[-5:])  # Last 5 issues
    
    def _enhance_feedback(self, feedback: CriticFeedback, plan: ExecutionPlan, results: List[StepResult]) -> CriticFeedback:
        """Enhance feedback với domain-specific knowledge."""
        # Kiểm tra medical safety concerns
        safety_issues = self._check_medical_safety(results)
        feedback.issues.extend(safety_issues)
        
        # Thêm specific suggestions based on performance
        performance_suggestions = self._generate_performance_suggestions(results)
        feedback.suggestions.extend(performance_suggestions)
        
        # Tính toán refined confidence
        feedback.confidence = self._calculate_refined_confidence(feedback, results)
        
        return feedback
    
    def _check_medical_safety(self, results: List[StepResult]) -> List[str]:
        """Kiểm tra safety concerns for medical domain."""
        safety_issues = []
        
        # Check for low confidence detections
        detector_result = next((r for r in results if "detect" in r.agent_name.lower()), None)
        if detector_result and detector_result.status.value == "success":
            objects = detector_result.data.get("objects", [])
            low_conf_objects = [obj for obj in objects if obj.get("confidence", 0) < 0.7]
            
            if low_conf_objects:
                safety_issues.append(f"LOW CONFIDENCE DETECTIONS: {len(low_conf_objects)} objects under 0.7 threshold")
        
        # Check for classification uncertainties
        for result in results:
            if "classifier" in result.agent_name.lower() and result.status.value == "success":
                conf = result.data.get("confidence", 0)
                if conf < 0.6:
                    safety_issues.append(f"UNCERTAIN CLASSIFICATION: {result.agent_name} confidence {conf:.2f}")
        
        return safety_issues
    
    def _generate_performance_suggestions(self, results: List[StepResult]) -> List[str]:
        """Tạo suggestions để cải thiện performance."""
        suggestions = []
        
        # Analyze processing times
        slow_steps = [r for r in results if r.processing_time > 10.0]
        if slow_steps:
            suggestions.append(f"OPTIMIZE PERFORMANCE: {len(slow_steps)} steps taking >10s")
        
        # Analyze error patterns
        failed_steps = [r for r in results if r.status.value == "failed"]
        if failed_steps:
            suggestions.append(f"FIX FAILURES: {len(failed_steps)} steps failed - check error logs")
        
        # Suggest ensemble methods for low confidence
        low_conf_classifiers = []
        for result in results:
            if "classifier" in result.agent_name.lower() and result.status.value == "success":
                if result.data.get("confidence", 0) < 0.8:
                    low_conf_classifiers.append(result.agent_name)
        
        if low_conf_classifiers:
            suggestions.append(f"ENSEMBLE CLASSIFICATION: Consider ensemble for {', '.join(low_conf_classifiers)}")
        
        return suggestions
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Tính toán confidence score cho feedback."""
        # Simple heuristic based on scores and issues
        overall_score = CriticScore[data.get("overall_score", "ACCEPTABLE").upper()]
        issues_count = len(data.get("issues", []))
        suggestions_count = len(data.get("suggestions", []))
        
        base_confidence = overall_score.value / 5.0
        issue_penalty = min(0.4, issues_count * 0.1)
        
        return max(0.1, base_confidence - issue_penalty)
    
    def _calculate_refined_confidence(self, feedback: CriticFeedback, results: List[StepResult]) -> float:
        """Tính toán refined confidence dựa trên chi tiết kết quả."""
        factors = []
        
        # Factor 1: Overall score
        factors.append(feedback.overall_score.value / 5.0)
        
        # Factor 2: Step scores consistency
        if feedback.step_scores:
            scores = [score.value for score in feedback.step_scores.values()]
            consistency = 1.0 - (np.std(scores) / 5.0) if scores else 0.5
            factors.append(consistency)
        
        # Factor 3: Issues vs suggestions ratio
        issue_ratio = len(feedback.issues) / max(1, len(feedback.issues) + len(feedback.suggestions))
        factors.append(1.0 - issue_ratio)
        
        # Factor 4: Success rate of steps
        success_rate = sum(1 for r in results if r.status.value == "success") / max(1, len(results))
        factors.append(success_rate)
        
        return np.mean(factors)
    
    def _create_fallback_feedback(self, plan_id: str, results: Optional[List[StepResult]] = None) -> CriticFeedback:
        """Tạo fallback feedback khi LLM fail."""
        feedback = CriticFeedback(
            plan_id=plan_id,
            overall_score=CriticScore.ACCEPTABLE,
            issues=["Unable to perform detailed evaluation due to system error"],
            suggestions=["Retry execution with increased timeouts"],
            reasoning="Fallback evaluation - full analysis unavailable",
            confidence=0.5
        )
        
        if results:
            # Basic analysis of results
            failed_count = sum(1 for r in results if r.status.value == "failed")
            if failed_count > 0:
                feedback.overall_score = CriticScore.POOR
                feedback.issues.append(f"{failed_count} steps failed execution")
        
        return feedback
    
    def _update_metrics(self, evaluation_time: float, score: CriticScore):
        """Cập nhật metrics."""
        self.metrics["total_evaluations"] += 1
        
        if score.value >= 4:
            self.metrics["high_quality_results"] += 1
        
        # Cập nhật average time
        total_time = self.metrics["average_evaluation_time"] * (self.metrics["total_evaluations"] - 1)
        self.metrics["average_evaluation_time"] = (total_time + evaluation_time) / self.metrics["total_evaluations"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Lấy metrics của Critic."""
        return {
            **self.metrics,
            "high_quality_rate": (
                self.metrics["high_quality_results"] / max(1, self.metrics["total_evaluations"])
            ) * 100
        }
    
    def analyze_trends(self, feedbacks: List[CriticFeedback]) -> Dict[str, Any]:
        """Phân tích trends từ historical feedbacks."""
        if not feedbacks:
            return {"status": "no_data"}
        
        # Overall score trends
        scores = [f.overall_score.value for f in feedbacks]
        score_trend = "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
        
        # Common issues
        all_issues = []
        for feedback in feedbacks:
            all_issues.extend(feedback.issues)
        
        from collections import Counter
        common_issues = Counter(all_issues).most_common(5)
        
        # Suggestion effectiveness (simplified)
        suggestion_count = sum(len(f.suggestions) for f in feedbacks)
        avg_suggestions = suggestion_count / len(feedbacks)
        
        return {
            "score_trend": score_trend,
            "average_score": np.mean(scores),
            "common_issues": [{"issue": issue, "count": count} for issue, count in common_issues],
            "average_suggestions_per_iteration": avg_suggestions,
            "total_iterations": len(feedbacks)
        }