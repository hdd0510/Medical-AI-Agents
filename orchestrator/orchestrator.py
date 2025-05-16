#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Medical Orchestrator with Reflexion Toggle
-------------------------------------------------
Single orchestrator có thể bật/tắt Reflexion mode.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import openai

from orchestrator.main import MedicalOrchestrator, OrchestratorConfig
from orchestrator.reflexion.planner import PlannerAgent
from orchestrator.reflexion.critic import CriticAgent
from orchestrator.reflexion.config import ReflexionConfig, ExecutionPlan, StepResult, CriticFeedback, CriticScore

@dataclass
class UnifiedOrchestratorConfig(OrchestratorConfig):
    """Config cho Unified Orchestrator."""
    # Reflexion toggle
    enable_reflexion: bool = False
    
    # Reflexion settings
    reflexion_max_iterations: int = 3
    reflexion_min_score_threshold: float = 3.0
    reflexion_timeout: float = 300.0
    
    # LLM settings
    reflexion_planner_model: str = "gpt-4"
    reflexion_critic_model: str = "gpt-4"
    reflexion_temperature: float = 0.7

class UnifiedMedicalOrchestrator:
    """
    Single orchestrator với Reflexion toggle.
    
    Usage:
        # Standard mode
        orch = UnifiedMedicalOrchestrator(config)
        result = orch.orchestrate(image, query)
        
        # Enable Reflexion
        orch.enable_reflexion()
        result = orch.orchestrate(image, query)  # Now uses Reflexion
        
        # Or force mode
        result = orch.orchestrate(image, query, use_reflexion=True)
    """
    
    def __init__(self, config: UnifiedOrchestratorConfig):
        self.config = config
        self.logger = logging.getLogger(f"unified.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.logging_level))
        
        # Always initialize base orchestrator
        self.base_orchestrator = MedicalOrchestrator(config)
        
        # Reflexion components (lazy init)
        self.planner = None
        self.critic = None
        self._reflexion_initialized = False
        
        # Stats
        self.stats = {
            "total_requests": 0,
            "standard_requests": 0,
            "reflexion_requests": 0
        }
        
        self.logger.info(f"Unified Orchestrator initialized (Reflexion: {'enabled' if config.enable_reflexion else 'disabled'})")
    
    def enable_reflexion(self, 
                        max_iterations: Optional[int] = None,
                        min_score: Optional[float] = None):
        """Enable Reflexion mode."""
        self.config.enable_reflexion = True
        
        if max_iterations:
            self.config.reflexion_max_iterations = max_iterations
        if min_score:
            self.config.reflexion_min_score_threshold = min_score
        
        self._ensure_reflexion_initialized()
        self.logger.info("Reflexion enabled")
    
    def disable_reflexion(self):
        """Disable Reflexion mode."""
        self.config.enable_reflexion = False
        self.logger.info("Reflexion disabled")
    
    def orchestrate(self, 
                   image_path: str, 
                   query: Optional[str] = None,
                   medical_context: Optional[Dict[str, Any]] = None,
                   use_reflexion: Optional[bool] = None) -> Dict[str, Any]:
        """
        Main orchestrate method.
        
        Args:
            image_path: Path to image
            query: User query
            medical_context: Medical context
            use_reflexion: Force use reflexion (overrides config)
            
        Returns:
            Results dict with orchestration mode info
        """
        self.stats["total_requests"] += 1
        
        # Determine mode
        if use_reflexion is not None:
            # Explicit override
            use_reflexion_mode = use_reflexion
        else:
            # Use config setting
            use_reflexion_mode = self.config.enable_reflexion
        
        # Run orchestration
        if use_reflexion_mode:
            self._ensure_reflexion_initialized()
            result = self._reflexion_orchestrate(image_path, query, medical_context)
            self.stats["reflexion_requests"] += 1
        else:
            result = self._standard_orchestrate(image_path, query, medical_context)
            self.stats["standard_requests"] += 1
        
        # Add mode info
        result["orchestration_mode"] = "reflexion" if use_reflexion_mode else "standard"
        result["reflexion_enabled"] = self.config.enable_reflexion
        
        return result
    
    def register_agents(self):
        """Register agents cho base orchestrator."""
        self.base_orchestrator.register_agents()
        self.logger.info("Agents registered")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        total = max(1, self.stats["total_requests"])  # Avoid division by zero
        
        return {
            **self.stats,
            "reflexion_usage_rate": (self.stats["reflexion_requests"] / total) * 100,
            "reflexion_enabled": self.config.enable_reflexion,
            "reflexion_config": {
                "max_iterations": self.config.reflexion_max_iterations,
                "min_score_threshold": self.config.reflexion_min_score_threshold
            }
        }
    
    def _ensure_reflexion_initialized(self):
        """Lazy initialization of Reflexion components."""
        if self._reflexion_initialized:
            return
        
        try:
            # Create Reflexion config
            reflexion_config = ReflexionConfig(
                max_iterations=self.config.reflexion_max_iterations,
                min_score_threshold=self.config.reflexion_min_score_threshold,
                total_timeout=self.config.reflexion_timeout,
                planner_model=self.config.reflexion_planner_model,
                critic_model=self.config.reflexion_critic_model,
                planner_temperature=self.config.reflexion_temperature
            )
            
            # Initialize components
            self.planner = PlannerAgent(reflexion_config)
            self.critic = CriticAgent(reflexion_config)
            
            self._reflexion_initialized = True
            self.logger.info("Reflexion components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Reflexion: {e}")
            raise
    
    def _standard_orchestrate(self, image_path: str, query: Optional[str], medical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard orchestration using base orchestrator."""
        self.logger.info("Running standard orchestration")
        result = self.base_orchestrator.orchestrate(image_path, query, medical_context)
        return result
    
    def _reflexion_orchestrate(self, image_path: str, query: Optional[str], medical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Reflexion orchestration - simplified version."""
        self.logger.info("Running Reflexion orchestration")
        start_time = time.time()
        
        try:
            iteration = 0
            best_result = None
            best_score = 0
            
            while iteration < self.config.reflexion_max_iterations:
                iteration += 1
                self.logger.info(f"Reflexion iteration {iteration}")
                
                # Create plan
                plan = self.planner.create_plan(
                    query=query or "",
                    image_path=image_path,
                    iteration_count=iteration - 1,
                    previous_feedback=[] if not best_result else [best_result.get("feedback", {}).get("issues", [])]
                )
                
                # Execute plan (simplified - use base orchestrator)
                step_result = self._execute_plan(plan, medical_context)
                
                # Evaluate with critic
                feedback = self.critic.evaluate_results(
                    plan=plan,
                    results=[step_result]
                )
                
                # Check if good enough
                current_score = feedback.overall_score.value
                if current_score >= self.config.reflexion_min_score_threshold:
                    self.logger.info(f"Reached acceptable score: {current_score}")
                    best_result = {
                        "result": step_result.data,
                        "feedback": {
                            "score": feedback.overall_score.name,
                            "issues": feedback.issues,
                            "suggestions": feedback.suggestions
                        }
                    }
                    break
                
                # Track best result
                if current_score > best_score:
                    best_score = current_score
                    best_result = {
                        "result": step_result.data,
                        "feedback": {
                            "score": feedback.overall_score.name,
                            "issues": feedback.issues,
                            "suggestions": feedback.suggestions
                        }
                    }
            
            total_time = time.time() - start_time
            
            # Format result
            if best_result:
                return {
                    **best_result["result"],
                    "success": True,
                    "iteration_count": iteration,
                    "final_score": best_result["feedback"]["score"],
                    "total_time": total_time,
                    "feedback": best_result["feedback"]
                }
            else:
                return {
                    "success": False,
                    "error": "All iterations failed",
                    "iteration_count": iteration,
                    "total_time": total_time
                }
                
        except Exception as e:
            self.logger.error(f"Reflexion orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "iteration_count": iteration,
                "total_time": time.time() - start_time
            }
    
    def _execute_plan(self, plan: ExecutionPlan, medical_context: Optional[Dict[str, Any]]) -> StepResult:
        """Execute plan using base orchestrator (simplified)."""
        try:
            # Use base orchestrator for execution
            result = self.base_orchestrator.orchestrate(
                image_path=plan.image_path,
                query=plan.query,
                medical_context=medical_context
            )
            
            return StepResult(
                step_id="unified_execution",
                status="success" if result.get("success") else "failed",
                data=result,
                processing_time=result.get("processing_time", 0),
                agent_name="unified_orchestrator"
            )
            
        except Exception as e:
            return StepResult(
                step_id="unified_execution", 
                status="failed",
                error=str(e),
                processing_time=0,
                agent_name="unified_orchestrator"
            )
    
    def compare_modes(self, image_path: str, query: str, medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compare Standard vs Reflexion modes."""
        self.logger.info("Comparing Standard vs Reflexion modes")
        
        # Run standard
        start_time = time.time()
        standard_result = self._standard_orchestrate(image_path, query, medical_context)
        standard_time = time.time() - start_time
        
        # Run reflexion (if enabled)
        if not self.config.enable_reflexion:
            self.logger.warning("Reflexion not enabled, cannot compare")
            return {
                "standard": {"result": standard_result, "time": standard_time},
                "reflexion": None,
                "error": "Reflexion not enabled"
            }
        
        start_time = time.time()
        reflexion_result = self._reflexion_orchestrate(image_path, query, medical_context)
        reflexion_time = time.time() - start_time
        
        return {
            "standard": {"result": standard_result, "time": standard_time},
            "reflexion": {"result": reflexion_result, "time": reflexion_time},
            "comparison": {
                "time_factor": reflexion_time / max(0.1, standard_time),
                "reflexion_iterations": reflexion_result.get("iteration_count", 0),
                "reflexion_score": reflexion_result.get("final_score", "N/A")
            }
        }

# Usage Examples:
"""
# Initialize
config = UnifiedOrchestratorConfig(
    enable_reflexion=False,  # Start with disabled
    reflexion_max_iterations=3
)
orch = UnifiedMedicalOrchestrator(config)
orch.register_agents()

# Standard mode
result1 = orch.orchestrate("image.jpg", "Is there a polyp?")
print(f"Mode: {result1['orchestration_mode']}")

# Enable Reflexion
orch.enable_reflexion(max_iterations=2, min_score=3.5)
result2 = orch.orchestrate("image.jpg", "Is there a polyp?")
print(f"Mode: {result2['orchestration_mode']}")

# Force specific mode
result3 = orch.orchestrate("image.jpg", "Question", use_reflexion=False)  # Force standard

# Get stats
stats = orch.get_stats()
print(f"Reflexion usage: {stats['reflexion_usage_rate']:.1f}%")

# Compare modes
comparison = orch.compare_modes("image.jpg", "Complex question")
"""