import os
import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import concurrent.futures

@dataclass
class OrchestratorConfig:
    """Configuration for the Medical Orchestrator."""
    name: str = "Medical Orchestrator"
    device: str = "cuda"
    parallel_execution: bool = True
    output_path: str = "results"
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    task_types: List[str] = field(default_factory=lambda: [
        "polyp_detection", "modality_classification", 
        "region_classification", "medical_qa", "comprehensive"
    ])

class MedicalOrchestrator:
    """
    Orchestrator for the Medical AI System with GPT-based agents.
    
    This orchestrator manages multiple specialized agents, each with reasoning 
    capabilities and specific tools. It determines which agents to use based on
    the task, coordinates their execution, and synthesizes the final result.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the Medical Orchestrator."""
        self.config = config
        self.logger = logging.getLogger(f"orchestrator.{self.config.name}")
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize agent registry
        self.agents = {}
        
        # Session ID for current orchestration
        self.current_session_id = None
        
        self.logger.info(f"Medical Orchestrator initialized with config: {config}")
    
    def _initialize_llm(self):
        """Initialize the LLM client for the orchestrator and agents."""
        try:
            import openai
            
            # Use API key from config or environment
            api_key = self.config.llm_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("No OpenAI API key found. Using mock LLM client.")
                return self._create_mock_llm()
                
            return openai.OpenAI(api_key=api_key)
            
        except ImportError:
            self.logger.warning("OpenAI package not installed. Using mock LLM client.")
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM client for testing."""
        from unittest.mock import MagicMock
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "reasoning": "Mock reasoning for testing",
            "answer": "Mock answer for testing"
        })
        mock_llm.chat.completions.create.return_value = mock_response
        
        return mock_llm
    
    def register_agent(self, name: str, agent):
        """Register an agent with the orchestrator."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    def orchestrate(self, 
                   image_path: str, 
                   query: Optional[str] = None, 
                   medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrate the processing of a medical image and query.
        
        Args:
            image_path: Path to the image
            query: Optional question or request
            medical_context: Optional medical context information
            
        Returns:
            Dictionary with orchestration results
        """
        start_time = time.time()
        
        # Create session ID
        self.current_session_id = str(uuid.uuid4())
        
        # Create output directory
        output_dir = os.path.join(self.config.output_path, self.current_session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.logger.info(f"Starting orchestration session {self.current_session_id}")
            self.logger.info(f"Image: {image_path}")
            self.logger.info(f"Query: {query or 'None'}")
            
            # 1. Analyze request and determine task type
            task_type, needed_agents = self._analyze_request(query)
            
            # 2. Execute agents based on task type
            results = {}
            
            if self.config.parallel_execution and task_type == "comprehensive":
                # Execute perception agents in parallel
                results.update(self._execute_agents_parallel(image_path, needed_agents))
            else:
                # Execute agents sequentially
                for agent_name in needed_agents:
                    agent_result = self._execute_agent(agent_name, image_path, query, medical_context, results)
                    results[agent_name] = agent_result
            
            # 3. Synthesize final results
            final_result = self._synthesize_results(task_type, results, query, medical_context)
            
            # Add metadata
            elapsed_time = time.time() - start_time
            final_result["metadata"] = {
                "session_id": self.current_session_id,
                "task_type": task_type,
                "processing_time": elapsed_time,
                "timestamp": time.time(),
                "query": query,
                "image_path": image_path
            }
            
            # Save results
            result_path = os.path.join(output_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Orchestration completed in {elapsed_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                "error": f"Orchestration failed: {str(e)}",
                "session_id": self.current_session_id,
                "success": False
            }
    
    def _analyze_request(self, query: Optional[str]) -> tuple:
        """
        Analyze the request and determine task type and needed agents.
        
        Args:
            query: Query from user
            
        Returns:
            tuple: (task_type, list_of_needed_agents)
        """
        # Default for no query: comprehensive analysis
        if not query:
            return "comprehensive", ["detector", "modality_classifier", "anatomical_classifier"]
        
        # Use LLM to analyze query
        if self.llm_client:
            try:
                prompt = f"""
                Analyze this medical endoscopy question and determine the most appropriate task type:
                
                Question: {query}
                
                Task types:
                - polyp_detection: Detecting polyps or abnormalities
                - modality_classification: Identifying imaging technique (WLI, BLI, etc.)
                - region_classification: Identifying anatomical location
                - medical_qa: Answering medical questions about the image
                - comprehensive: Requiring multiple analyses
                
                Respond with ONLY the task type and the list of needed agents as JSON:
                {{
                    "task_type": "one_of_the_task_types_above",
                    "needed_agents": ["agent1", "agent2"]
                }}
                
                Available agents:
                - detector: For polyp detection
                - modality_classifier: For imaging technique classification
                - anatomical_classifier: For anatomical location classification
                - vqa: For medical question answering
                """
                
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                
                # Extract and parse JSON from response
                content = response.choices[0].message.content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    analysis = json.loads(json_content)
                    task_type = analysis.get("task_type")
                    needed_agents = analysis.get("needed_agents", [])
                    
                    # Validate task type
                    if task_type in self.config.task_types:
                        self.logger.info(f"LLM determined task type: {task_type}, needed agents: {needed_agents}")
                        return task_type, needed_agents
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze request with LLM: {e}")
        
        # Fallback: Simple keyword-based classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["polyp", "detect", "find", "see", "abnormality"]):
            return "polyp_detection", ["detector"]
        
        if any(word in query_lower for word in ["technique", "modality", "imaging", "wli", "bli"]):
            return "modality_classification", ["modality_classifier"]
        
        if any(word in query_lower for word in ["location", "region", "where", "stomach", "colon"]):
            return "region_classification", ["anatomical_classifier"]
        
        if "?" in query:
            return "medical_qa", ["detector", "vqa"]
        
        # Default: comprehensive
        return "comprehensive", ["detector", "modality_classifier", "anatomical_classifier", "vqa"]
    
    def _execute_agents_parallel(self, image_path: str, agent_names: List[str]) -> Dict[str, Any]:
        """Execute multiple agents in parallel."""
        results = {}
        valid_agents = [name for name in agent_names if name in self.agents]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures for each agent
            future_to_agent = {
                executor.submit(self._execute_agent, agent_name, image_path, None, None, {}): agent_name
                for agent_name in valid_agents
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                except Exception as e:
                    self.logger.error(f"Error executing agent {agent_name}: {e}")
                    results[agent_name] = {"error": str(e), "success": False}
        
        return results
    
    def _execute_agent(self, 
                      agent_name: str, 
                      image_path: str, 
                      query: Optional[str], 
                      medical_context: Optional[Dict[str, Any]],
                      previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return {"error": f"Agent {agent_name} not found", "success": False}
        
        try:
            # Prepare input data
            input_data = {
                "image_path": image_path,
                "previous_results": previous_results
            }
            
            # Add query if provided
            if query:
                input_data["query"] = query
            
            # Add medical context if provided
            if medical_context:
                input_data["medical_context"] = medical_context
            
            # Process with agent
            self.logger.info(f"Executing agent: {agent_name}")
            result = agent.process(input_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in agent {agent_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {"error": str(e), "success": False}
    
    def _synthesize_results(self, 
                           task_type: str, 
                           results: Dict[str, Any], 
                           query: Optional[str],
                           medical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents into a final result.
        
        Args:
            task_type: Type of task being performed
            results: Results from individual agents
            query: Original query
            medical_context: Medical context
            
        Returns:
            Synthesized results
        """
        # Base structure for final result
        final_result = {
            "task_type": task_type,
            "success": True
        }
        
        # Determine synthesis approach based on task type
        if task_type == "polyp_detection":
            # For polyp detection, use detector results directly
            if "detector" in results:
                detector_result = results["detector"]
                final_result.update({
                    "detections": detector_result.get("detections", []),
                    "detection_count": len(detector_result.get("detections", [])),
                    "confidence": detector_result.get("confidence", 0.0),
                    "answer": detector_result.get("answer", "")
                })
                
        elif task_type == "modality_classification":
            # For modality classification, use modality classifier results
            if "modality_classifier" in results:
                modal_result = results["modality_classifier"]
                final_result.update({
                    "modality": modal_result.get("class", "Unknown"),
                    "confidence": modal_result.get("confidence", 0.0),
                    "answer": modal_result.get("answer", "")
                })
                
        elif task_type == "region_classification":
            # For region classification, use anatomical classifier results
            if "anatomical_classifier" in results:
                region_result = results["anatomical_classifier"]
                final_result.update({
                    "region": region_result.get("class", "Unknown"),
                    "confidence": region_result.get("confidence", 0.0),
                    "answer": region_result.get("answer", "")
                })
                
        elif task_type == "medical_qa":
            # For medical QA, prioritize VQA results
            if "vqa" in results:
                vqa_result = results["vqa"]
                final_result.update({
                    "answer": vqa_result.get("answer", ""),
                    "confidence": vqa_result.get("confidence", 0.0)
                })
            
            # Add supporting data from detector if available
            if "detector" in results:
                detector_result = results["detector"]
                final_result.update({
                    "detections": detector_result.get("detections", []),
                    "detection_count": len(detector_result.get("detections", []))
                })
                
        elif task_type == "comprehensive":
            # For comprehensive, combine all results and generate a summary
            
            # Include detector results
            if "detector" in results:
                detector_result = results["detector"]
                final_result.update({
                    "detections": detector_result.get("detections", []),
                    "detection_count": len(detector_result.get("detections", []))
                })
            
            # Include modality classification
            if "modality_classifier" in results:
                modal_result = results["modality_classifier"]
                final_result.update({
                    "modality": modal_result.get("class", "Unknown"),
                    "modality_confidence": modal_result.get("confidence", 0.0)
                })
            
            # Include region classification
            if "anatomical_classifier" in results:
                region_result = results["anatomical_classifier"]
                final_result.update({
                    "region": region_result.get("class", "Unknown"),
                    "region_confidence": region_result.get("confidence", 0.0)
                })
            
            # Include VQA answer if query was provided
            if query and "vqa" in results:
                vqa_result = results["vqa"]
                final_result.update({
                    "answer": vqa_result.get("answer", ""),
                    "answer_confidence": vqa_result.get("confidence", 0.0)
                })
            
            # Generate a summary
            final_result["summary"] = self._generate_summary(final_result, query)
        
        return final_result
    
    def _generate_summary(self, result: Dict[str, Any], query: Optional[str]) -> str:
        """Generate a summary from the comprehensive results."""
        summary_parts = []
        
        # Summarize detection results
        if "detection_count" in result:
            count = result["detection_count"]
            if count > 0:
                summary_parts.append(f"Found {count} polyps or abnormalities in the image.")
            else:
                summary_parts.append("No polyps or abnormalities detected.")
        
        # Summarize modality
        if "modality" in result:
            modality = result["modality"]
            confidence = result.get("modality_confidence", 0.0)
            if confidence > 0.7:
                summary_parts.append(f"Image was captured using {modality} imaging technique.")
            else:
                summary_parts.append(f"Image likely uses {modality} imaging technique (moderate confidence).")
        
        # Summarize anatomical region
        if "region" in result:
            region = result["region"]
            confidence = result.get("region_confidence", 0.0)
            if confidence > 0.7:
                summary_parts.append(f"The image shows the {region} region of the GI tract.")
            else:
                summary_parts.append(f"The image may show the {region} region (moderate confidence).")
        
        # Include answer to query if available
        if query and "answer" in result:
            summary_parts.append(f"Response to query: {result['answer']}")
        
        # Join all parts
        return " ".join(summary_parts)