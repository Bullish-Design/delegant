"""
Delegant Workflow Decorators
============================

Decorator-based patterns for chaining, routing, and orchestrating multiple agents.
Provides high-level workflow patterns for complex multi-agent operations.
"""

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime
import json

from ..agent import Agent
from ..exceptions import WorkflowExecutionError, ValidationError
from ..config import get_config

logger = logging.getLogger(__name__)


class WorkflowResult:
    """Result from workflow execution with comprehensive metadata."""
    
    def __init__(
        self,
        workflow_type: str,
        success: bool,
        results: Dict[str, Any],
        execution_time: float,
        agent_states: Dict[str, str],
        errors: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.workflow_type = workflow_type
        self.success = success
        self.results = results
        self.execution_time = execution_time
        self.agent_states = agent_states
        self.errors = errors or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_type": self.workflow_type,
            "success": self.success,
            "results": self.results,
            "execution_time": self.execution_time,
            "agent_states": self.agent_states,
            "errors": self.errors,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"Workflow {self.workflow_type}: {status} ({self.execution_time:.2f}s)"


class WorkflowExecutor:
    """Base class for workflow execution with common functionality."""
    
    def __init__(self, workflow_type: str):
        self.workflow_type = workflow_type
        self.execution_history: List[WorkflowResult] = []
    
    async def _execute_agent_method(
        self, 
        agent: Agent, 
        method_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute a method on an agent with error handling."""
        try:
            method = getattr(agent, method_name)
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Agent method execution failed: {agent.__class__.__name__}.{method_name} - {e}")
            raise
    
    def _get_agent_state(self, agent: Agent) -> str:
        """Get current state of an agent."""
        try:
            status = agent.get_server_status()
            connected_servers = sum(1 for s in status.values() if s.get('status') == 'connected')
            total_servers = len(status)
            return f"connected_{connected_servers}/{total_servers}"
        except:
            return "unknown"
    
    def _record_execution(self, result: WorkflowResult) -> None:
        """Record workflow execution in history."""
        self.execution_history.append(result)
        
        # Limit history size
        max_history = get_config().connection_pool_size  # Reuse config value
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]


class ChainExecutor(WorkflowExecutor):
    """Executor for sequential agent chains."""
    
    def __init__(self, agents: List[Type[Agent]]):
        super().__init__("chain")
        self.agent_classes = agents
        self.agents: List[Agent] = []
    
    async def __aenter__(self):
        """Initialize agents for the chain."""
        for agent_class in self.agent_classes:
            agent = agent_class()
            await agent.__aenter__()
            self.agents.append(agent)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup agents."""
        for agent in self.agents:
            try:
                await agent.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error cleaning up agent {agent.__class__.__name__}: {e}")
    
    async def execute(self, method_name: str, *args, **kwargs) -> WorkflowResult:
        """Execute method on all agents in sequence."""
        start_time = datetime.now()
        results = {}
        errors = {}
        agent_states = {}
        
        try:
            current_input = (args, kwargs)
            
            for i, agent in enumerate(self.agents):
                agent_name = f"{agent.__class__.__name__}_{i}"
                agent_states[agent_name] = self._get_agent_state(agent)
                
                try:
                    # Execute method on agent
                    if i == 0:
                        # First agent gets original input
                        result = await self._execute_agent_method(agent, method_name, *args, **kwargs)
                    else:
                        # Subsequent agents get output from previous agent
                        if isinstance(current_input, tuple) and len(current_input) == 2:
                            prev_args, prev_kwargs = current_input
                            # Pass previous result as first argument
                            result = await self._execute_agent_method(
                                agent, method_name, results[list(results.keys())[-1]], **prev_kwargs
                            )
                        else:
                            result = await self._execute_agent_method(agent, method_name, current_input)
                    
                    results[agent_name] = result
                    current_input = result
                    
                except Exception as e:
                    errors[agent_name] = str(e)
                    # Chain breaks on first error
                    break
            
            execution_time = (datetime.now() - start_time).total_seconds()
            success = len(errors) == 0
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=success,
                results=results,
                execution_time=execution_time,
                agent_states=agent_states,
                errors=errors,
                metadata={
                    "chain_length": len(self.agents),
                    "completed_steps": len(results),
                    "method_name": method_name
                }
            )
            
            self._record_execution(workflow_result)
            return workflow_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=False,
                results=results,
                execution_time=execution_time,
                agent_states=agent_states,
                errors={"workflow": str(e)},
                metadata={"method_name": method_name}
            )
            
            self._record_execution(workflow_result)
            raise WorkflowExecutionError(
                workflow_type=self.workflow_type,
                failed_step="chain_execution",
                partial_results=results,
                agent_states=agent_states,
                original_error=e
            )


class RouterExecutor(WorkflowExecutor):
    """Executor for routing requests to appropriate agents."""
    
    def __init__(self, agent_map: Dict[str, Type[Agent]]):
        super().__init__("router")
        self.agent_map = agent_map
        self.agents: Dict[str, Agent] = {}
        self.routing_history: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        """Initialize agents for routing."""
        for name, agent_class in self.agent_map.items():
            agent = agent_class()
            await agent.__aenter__()
            self.agents[name] = agent
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup agents."""
        for agent in self.agents.values():
            try:
                await agent.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")
    
    def _route_request(self, *args, **kwargs) -> str:
        """Determine which agent should handle the request."""
        # Simple routing logic - can be overridden
        # Look for routing hints in kwargs
        if 'route_to' in kwargs:
            route_to = kwargs.pop('route_to')
            if route_to in self.agents:
                return route_to
        
        # Default to first agent if no routing specified
        return list(self.agents.keys())[0]
    
    async def execute(self, method_name: str, *args, **kwargs) -> WorkflowResult:
        """Route request to appropriate agent and execute."""
        start_time = datetime.now()
        
        try:
            # Determine routing
            selected_agent_name = self._route_request(*args, **kwargs)
            
            if selected_agent_name not in self.agents:
                raise ValidationError(
                    validation_target="routing_decision",
                    field_errors={"agent": f"Agent '{selected_agent_name}' not found"},
                    invalid_value=selected_agent_name
                )
            
            selected_agent = self.agents[selected_agent_name]
            
            # Record routing decision
            routing_decision = {
                "selected_agent": selected_agent_name,
                "available_agents": list(self.agents.keys()),
                "method_name": method_name,
                "timestamp": datetime.now().isoformat()
            }
            self.routing_history.append(routing_decision)
            
            # Execute on selected agent
            result = await self._execute_agent_method(selected_agent, method_name, *args, **kwargs)
            
            # Get agent states
            agent_states = {name: self._get_agent_state(agent) for name, agent in self.agents.items()}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=True,
                results={selected_agent_name: result},
                execution_time=execution_time,
                agent_states=agent_states,
                metadata={
                    "routing_decision": routing_decision,
                    "method_name": method_name
                }
            )
            
            self._record_execution(workflow_result)
            return workflow_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=False,
                results={},
                execution_time=execution_time,
                agent_states={name: self._get_agent_state(agent) for name, agent in self.agents.items()},
                errors={"routing": str(e)},
                metadata={"method_name": method_name}
            )
            
            self._record_execution(workflow_result)
            raise WorkflowExecutionError(
                workflow_type=self.workflow_type,
                failed_step="routing_execution",
                agent_states={name: self._get_agent_state(agent) for name, agent in self.agents.items()},
                original_error=e
            )


class ParallelExecutor(WorkflowExecutor):
    """Executor for parallel agent execution."""
    
    def __init__(self, agents: List[Type[Agent]]):
        super().__init__("parallel")
        self.agent_classes = agents
        self.agents: List[Agent] = []
    
    async def __aenter__(self):
        """Initialize agents for parallel execution."""
        for agent_class in self.agent_classes:
            agent = agent_class()
            await agent.__aenter__()
            self.agents.append(agent)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup agents."""
        for agent in self.agents:
            try:
                await agent.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error cleaning up agent {agent.__class__.__name__}: {e}")
    
    async def execute(self, method_name: str, *args, **kwargs) -> WorkflowResult:
        """Execute method on all agents in parallel."""
        start_time = datetime.now()
        
        try:
            # Create tasks for parallel execution
            tasks = []
            agent_names = []
            
            for i, agent in enumerate(self.agents):
                agent_name = f"{agent.__class__.__name__}_{i}"
                agent_names.append(agent_name)
                
                task = asyncio.create_task(
                    self._execute_agent_method(agent, method_name, *args, **kwargs),
                    name=agent_name
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            results = {}
            errors = {}
            
            for i, (agent_name, result) in enumerate(zip(agent_names, results_list)):
                if isinstance(result, Exception):
                    errors[agent_name] = str(result)
                else:
                    results[agent_name] = result
            
            # Get agent states
            agent_states = {
                agent_names[i]: self._get_agent_state(agent) 
                for i, agent in enumerate(self.agents)
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            success = len(errors) == 0
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=success,
                results=results,
                execution_time=execution_time,
                agent_states=agent_states,
                errors=errors,
                metadata={
                    "parallel_count": len(self.agents),
                    "successful_count": len(results),
                    "failed_count": len(errors),
                    "method_name": method_name
                }
            )
            
            self._record_execution(workflow_result)
            return workflow_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=False,
                results={},
                execution_time=execution_time,
                agent_states={f"{agent.__class__.__name__}_{i}": self._get_agent_state(agent) 
                             for i, agent in enumerate(self.agents)},
                errors={"parallel_execution": str(e)},
                metadata={"method_name": method_name}
            )
            
            self._record_execution(workflow_result)
            raise WorkflowExecutionError(
                workflow_type=self.workflow_type,
                failed_step="parallel_execution",
                agent_states={f"{agent.__class__.__name__}_{i}": self._get_agent_state(agent) 
                             for i, agent in enumerate(self.agents)},
                original_error=e
            )


class OrchestratorExecutor(WorkflowExecutor):
    """Executor for complex multi-agent orchestration with conditional logic."""
    
    def __init__(self, agent_map: Dict[str, Type[Agent]]):
        super().__init__("orchestrator")
        self.agent_map = agent_map
        self.agents: Dict[str, Agent] = {}
        self.orchestration_plan: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        """Initialize agents for orchestration."""
        for name, agent_class in self.agent_map.items():
            agent = agent_class()
            await agent.__aenter__()
            self.agents[name] = agent
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup agents."""
        for agent in self.agents.values():
            try:
                await agent.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")
    
    def add_step(
        self, 
        agent_name: str, 
        method_name: str, 
        condition: Optional[Callable] = None,
        depends_on: Optional[List[str]] = None,
        parallel_group: Optional[str] = None
    ) -> None:
        """Add a step to the orchestration plan."""
        step = {
            "agent_name": agent_name,
            "method_name": method_name,
            "condition": condition,
            "depends_on": depends_on or [],
            "parallel_group": parallel_group,
            "step_id": f"step_{len(self.orchestration_plan)}"
        }
        self.orchestration_plan.append(step)
    
    async def execute(self, *args, **kwargs) -> WorkflowResult:
        """Execute the orchestration plan."""
        start_time = datetime.now()
        
        try:
            results = {}
            errors = {}
            completed_steps = set()
            step_results = {}
            
            # Group steps by parallel groups
            parallel_groups = {}
            sequential_steps = []
            
            for step in self.orchestration_plan:
                if step["parallel_group"]:
                    if step["parallel_group"] not in parallel_groups:
                        parallel_groups[step["parallel_group"]] = []
                    parallel_groups[step["parallel_group"]].append(step)
                else:
                    sequential_steps.append(step)
            
            # Execute steps
            all_steps = sequential_steps.copy()
            
            # Add parallel groups as single units
            for group_name, group_steps in parallel_groups.items():
                all_steps.append({
                    "type": "parallel_group",
                    "group_name": group_name,
                    "steps": group_steps
                })
            
            for step in all_steps:
                if step.get("type") == "parallel_group":
                    # Execute parallel group
                    group_results = await self._execute_parallel_group(
                        step["steps"], step_results, *args, **kwargs
                    )
                    step_results.update(group_results)
                else:
                    # Execute single step
                    step_result = await self._execute_single_step(
                        step, step_results, *args, **kwargs
                    )
                    if step_result is not None:
                        step_results[step["step_id"]] = step_result
                        completed_steps.add(step["step_id"])
            
            # Compile final results
            for step in self.orchestration_plan:
                agent_name = step["agent_name"]
                step_id = step["step_id"]
                
                if step_id in step_results:
                    if agent_name not in results:
                        results[agent_name] = {}
                    results[agent_name][step["method_name"]] = step_results[step_id]
            
            # Get agent states
            agent_states = {name: self._get_agent_state(agent) for name, agent in self.agents.items()}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            success = len(errors) == 0
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=success,
                results=results,
                execution_time=execution_time,
                agent_states=agent_states,
                errors=errors,
                metadata={
                    "orchestration_plan": len(self.orchestration_plan),
                    "completed_steps": len(completed_steps),
                    "parallel_groups": len(parallel_groups)
                }
            )
            
            self._record_execution(workflow_result)
            return workflow_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            workflow_result = WorkflowResult(
                workflow_type=self.workflow_type,
                success=False,
                results={},
                execution_time=execution_time,
                agent_states={name: self._get_agent_state(agent) for name, agent in self.agents.items()},
                errors={"orchestration": str(e)},
                metadata={"orchestration_plan": len(self.orchestration_plan)}
            )
            
            self._record_execution(workflow_result)
            raise WorkflowExecutionError(
                workflow_type=self.workflow_type,
                failed_step="orchestration_execution",
                agent_states={name: self._get_agent_state(agent) for name, agent in self.agents.items()},
                original_error=e
            )
    
    async def _execute_single_step(
        self, 
        step: Dict[str, Any], 
        step_results: Dict[str, Any],
        *args, 
        **kwargs
    ) -> Any:
        """Execute a single orchestration step."""
        # Check dependencies
        for dep in step["depends_on"]:
            if dep not in step_results:
                logger.warning(f"Step {step['step_id']} skipped due to missing dependency: {dep}")
                return None
        
        # Check condition
        if step["condition"] and not step["condition"](step_results):
            logger.info(f"Step {step['step_id']} skipped due to condition")
            return None
        
        # Execute step
        agent = self.agents[step["agent_name"]]
        
        # Prepare arguments (include previous results if available)
        step_args = args
        step_kwargs = kwargs.copy()
        
        # Add dependency results to kwargs
        for dep in step["depends_on"]:
            if dep in step_results:
                step_kwargs[f"dep_{dep}"] = step_results[dep]
        
        return await self._execute_agent_method(agent, step["method_name"], *step_args, **step_kwargs)
    
    async def _execute_parallel_group(
        self, 
        steps: List[Dict[str, Any]], 
        step_results: Dict[str, Any],
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a group of steps in parallel."""
        tasks = []
        step_ids = []
        
        for step in steps:
            step_ids.append(step["step_id"])
            task = asyncio.create_task(
                self._execute_single_step(step, step_results, *args, **kwargs),
                name=step["step_id"]
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        group_results = {}
        for step_id, result in zip(step_ids, results_list):
            if not isinstance(result, Exception) and result is not None:
                group_results[step_id] = result
        
        return group_results


# Decorator functions

def chain(*agents: Type[Agent]) -> Callable:
    """Decorator for chaining agents in sequential execution.
    
    Args:
        *agents: Agent classes to chain together
        
    Returns:
        Decorator function
        
    Example:
        @chain(SearchAgent, AnalysisAgent, ReportAgent)
        class ResearchPipeline(Agent):
            async def research(self, topic: str) -> dict:
                return await self.execute_chain("research", topic)
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self._chain_executor = None
            self._chain_agents = agents
        
        async def execute_chain(self, method_name: str, *args, **kwargs):
            """Execute the agent chain."""
            if self._chain_executor is None:
                self._chain_executor = ChainExecutor(self._chain_agents)
            
            async with self._chain_executor as executor:
                return await executor.execute(method_name, *args, **kwargs)
        
        cls.__init__ = new_init
        cls.execute_chain = execute_chain
        return cls
    
    return decorator


def router(agent_map: Dict[str, Type[Agent]]) -> Callable:
    """Decorator for routing requests to appropriate agents.
    
    Args:
        agent_map: Dictionary mapping route names to agent classes
        
    Returns:
        Decorator function
        
    Example:
        @router({"search": SearchAgent, "analysis": AnalysisAgent})
        class SmartRouter(Agent):
            async def process(self, request: str, route_to: str) -> dict:
                return await self.execute_route("process", request, route_to=route_to)
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self._router_executor = None
            self._agent_map = agent_map
        
        async def execute_route(self, method_name: str, *args, **kwargs):
            """Execute via routing."""
            if self._router_executor is None:
                self._router_executor = RouterExecutor(self._agent_map)
            
            async with self._router_executor as executor:
                return await executor.execute(method_name, *args, **kwargs)
        
        cls.__init__ = new_init
        cls.execute_route = execute_route
        return cls
    
    return decorator


def parallel(*agents: Type[Agent]) -> Callable:
    """Decorator for parallel agent execution.
    
    Args:
        *agents: Agent classes to execute in parallel
        
    Returns:
        Decorator function
        
    Example:
        @parallel(SearchAgent, NewsAgent, SocialAgent)
        class ParallelResearch(Agent):
            async def gather_data(self, topic: str) -> dict:
                return await self.execute_parallel("search", topic)
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self._parallel_executor = None
            self._parallel_agents = agents
        
        async def execute_parallel(self, method_name: str, *args, **kwargs):
            """Execute agents in parallel."""
            if self._parallel_executor is None:
                self._parallel_executor = ParallelExecutor(self._parallel_agents)
            
            async with self._parallel_executor as executor:
                return await executor.execute(method_name, *args, **kwargs)
        
        cls.__init__ = new_init
        cls.execute_parallel = execute_parallel
        return cls
    
    return decorator


def orchestrator(agent_map: Dict[str, Type[Agent]]) -> Callable:
    """Decorator for complex multi-agent orchestration.
    
    Args:
        agent_map: Dictionary mapping agent names to agent classes
        
    Returns:
        Decorator function
        
    Example:
        @orchestrator({"search": SearchAgent, "analysis": AnalysisAgent, "report": ReportAgent})
        class ComplexWorkflow(Agent):
            def setup_orchestration(self):
                self._orchestrator_executor.add_step("search", "search", depends_on=[])
                self._orchestrator_executor.add_step("analysis", "analyze", depends_on=["step_0"])
                self._orchestrator_executor.add_step("report", "generate", depends_on=["step_1"])
            
            async def execute_workflow(self, topic: str) -> dict:
                self.setup_orchestration()
                return await self.execute_orchestration(topic)
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self._orchestrator_executor = None
            self._orchestrator_agent_map = agent_map
        
        async def execute_orchestration(self, *args, **kwargs):
            """Execute orchestrated workflow."""
            if self._orchestrator_executor is None:
                self._orchestrator_executor = OrchestratorExecutor(self._orchestrator_agent_map)
            
            async with self._orchestrator_executor as executor:
                return await executor.execute(*args, **kwargs)
        
        def add_orchestration_step(self, agent_name: str, method_name: str, **step_kwargs):
            """Add step to orchestration plan."""
            if self._orchestrator_executor is None:
                self._orchestrator_executor = OrchestratorExecutor(self._orchestrator_agent_map)
            self._orchestrator_executor.add_step(agent_name, method_name, **step_kwargs)
        
        cls.__init__ = new_init
        cls.execute_orchestration = execute_orchestration
        cls.add_orchestration_step = add_orchestration_step
        return cls
    
    return decorator


# Example usage and testing
if __name__ == "__main__":
    from ..agent import Agent
    from ..servers.filesystem import FileSystemServer
    from ..servers.websearch import WebSearchServer
    
    class SearchAgent(Agent):
        """Agent for web search."""
        instruction: str = "Search the web for information"
        search: WebSearchServer = WebSearchServer(provider="duckduckgo")
        
        async def search_topic(self, topic: str) -> dict:
            results = await self.search.search(topic, max_results=5)
            return {"topic": topic, "results": results}
    
    class AnalysisAgent(Agent):
        """Agent for analysis."""
        instruction: str = "Analyze search results"
        files: FileSystemServer = FileSystemServer()
        
        async def analyze_results(self, search_data: dict) -> dict:
            # Simulate analysis
            return {
                "topic": search_data["topic"],
                "analysis": f"Analyzed {len(search_data.get('results', []))} results",
                "summary": "Analysis complete"
            }
    
    @chain(SearchAgent, AnalysisAgent)
    class ResearchPipeline(Agent):
        """Research pipeline using chain decorator."""
        instruction: str = "Complete research pipeline"
        
        async def research(self, topic: str) -> dict:
            return await self.execute_chain("search_topic", topic)
    
    @parallel(SearchAgent, AnalysisAgent)
    class ParallelResearch(Agent):
        """Parallel research using parallel decorator."""
        instruction: str = "Parallel research execution"
        
        async def gather_data(self, topic: str) -> dict:
            return await self.execute_parallel("search_topic", topic)
    
    async def test_workflows():
        # Test chain workflow
        print("Testing chain workflow...")
        async with ResearchPipeline() as pipeline:
            result = await pipeline.research("artificial intelligence")
            print(f"Chain result: {result.success}")
        
        # Test parallel workflow  
        print("Testing parallel workflow...")
        async with ParallelResearch() as parallel:
            result = await parallel.gather_data("machine learning")
            print(f"Parallel result: {result.success}")
    
    # Run test
    # asyncio.run(test_workflows())
