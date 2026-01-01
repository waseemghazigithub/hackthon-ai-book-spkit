---
sidebar_position: 3
---

# LLM-Based Task Planning

## Introduction

Large Language Models (LLMs) have revolutionized how we approach task planning for robotic systems. By leveraging the reasoning capabilities of LLMs, we can create sophisticated task planners that decompose high-level human instructions into executable robotic actions. This chapter explores how to integrate LLMs with robotic systems for intelligent task planning and execution.

## LLM Task Planning Architecture

### Overview

LLM-based task planning involves several key components working together:

1. **Natural Language Interface**: Accepts high-level human instructions
2. **Task Decomposition**: Breaks down complex tasks into simpler subtasks
3. **Action Mapping**: Maps subtasks to specific robotic actions
4. **Execution Monitoring**: Tracks task progress and handles failures
5. **Feedback Integration**: Incorporates execution results back into planning

### System Components

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import openai
import json
import re

class LLMTaskPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_task_planner')

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key="your-api-key-here")

        # Publishers and subscribers
        self.task_request_sub = self.create_subscription(
            String, 'task_request', self.task_request_callback, 10)
        self.task_status_pub = self.create_publisher(
            String, 'task_status', 10)

        # Action clients for robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Robot capabilities database
        self.robot_capabilities = {
            "navigation": ["go to", "move to", "navigate to", "reach"],
            "manipulation": ["pick up", "grasp", "grab", "place", "put"],
            "perception": ["find", "locate", "identify", "recognize"],
            "interaction": ["greet", "introduce", "follow", "wait"]
        }

        # Environment map
        self.location_map = {
            "kitchen": (3.0, 2.0, 0.0),
            "living room": (0.0, 0.0, 0.0),
            "bedroom": (-2.0, 3.0, 1.57),
            "office": (4.0, -1.0, -1.57),
            "dining room": (1.0, -2.0, 0.0)
        }

    def task_request_callback(self, msg):
        """Handle incoming task requests"""
        task_description = msg.data
        self.get_logger().info(f'Received task: {task_description}')

        # Plan the task using LLM
        plan = self.plan_task_with_llm(task_description)

        if plan:
            self.execute_plan(plan)
        else:
            self.get_logger().error('Failed to generate plan')

    def plan_task_with_llm(self, task_description):
        """Generate task plan using LLM"""
        prompt = f"""
        You are a task planner for a humanoid robot. Given the following task, break it down into executable steps.
        Consider the robot's capabilities: {list(self.robot_capabilities.keys())}
        Available locations: {list(self.location_map.keys())}

        Task: {task_description}

        Return the plan as a JSON array of steps, where each step has:
        - "action": the action to perform
        - "target": the target of the action
        - "location": where to perform the action (if applicable)
        - "description": human-readable description

        Example:
        [
            {{"action": "navigate", "target": "kitchen", "location": "kitchen", "description": "Go to the kitchen"}},
            {{"action": "perceive", "target": "water bottle", "location": "kitchen", "description": "Look for a water bottle"}},
            {{"action": "manipulate", "target": "water bottle", "location": "kitchen", "description": "Pick up the water bottle"}}
        ]

        Return only the JSON array, nothing else:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            plan_text = response.choices[0].message.content.strip()

            # Extract JSON from response if it contains other text
            json_match = re.search(r'\[.*\]', plan_text, re.DOTALL)
            if json_match:
                plan_json = json_match.group()
                plan = json.loads(plan_json)
                return plan
            else:
                self.get_logger().error(f'Could not extract JSON from LLM response: {plan_text}')
                return None

        except Exception as e:
            self.get_logger().error(f'LLM planning failed: {e}')
            return None
```

## Task Decomposition Strategies

### Hierarchical Task Networks (HTN)

LLMs can generate hierarchical task networks that break down complex tasks:

```python
class HierarchicalTaskPlanner:
    def __init__(self):
        self.task_library = {
            "make_coffee": [
                {"action": "navigate", "target": "kitchen"},
                {"action": "find", "target": "coffee_maker"},
                {"action": "find", "target": "coffee_beans"},
                {"action": "operate", "target": "coffee_maker"}
            ],
            "clean_room": [
                {"action": "navigate", "target": "bedroom"},
                {"action": "find", "target": "dirt"},
                {"action": "navigate", "target": "waste_bin"},
                {"action": "place", "target": "dirt"}
            ],
            "greet_guest": [
                {"action": "navigate", "target": "entrance"},
                {"action": "find", "target": "person"},
                {"action": "greet", "target": "person"}
            ]
        }

    def decompose_task(self, task_description, llm_client):
        """Decompose task using LLM with hierarchical knowledge"""
        prompt = f"""
        Decompose the following task into subtasks. Use the provided task library as examples:
        {json.dumps(self.task_library, indent=2)}

        Task to decompose: {task_description}

        Return a hierarchical plan as JSON with 'name', 'type', 'subtasks' (if any), and 'primitive_actions' fields.
        """

        response = llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.choices[0].message.content)
```

### Context-Aware Planning

```python
class ContextAwarePlanner:
    def __init__(self):
        self.context = {
            "current_time": "morning",
            "current_location": "living room",
            "battery_level": 0.8,
            "weather": "sunny",
            "human_activities": ["cooking", "working"]
        }

    def generate_contextual_plan(self, task, context, llm_client):
        """Generate plan considering current context"""
        prompt = f"""
        Generate a plan for: {task}

        Current context:
        - Time: {context.get('current_time', 'unknown')}
        - Location: {context.get('current_location', 'unknown')}
        - Battery: {context.get('battery_level', 'unknown')}
        - Weather: {context.get('weather', 'unknown')}
        - Human activities: {context.get('human_activities', [])}

        Consider the context when generating the plan. For example:
        - If battery is low, prioritize efficiency
        - If weather is bad, avoid outdoor tasks
        - If humans are busy, avoid interrupting

        Return the plan as JSON with actions, priorities, and justifications.
        """

        response = llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.choices[0].message.content)
```

## Action Mapping and Execution

### Mapping LLM Output to Robot Actions

```python
class ActionMapper:
    def __init__(self):
        # Define action mappings
        self.action_mapping = {
            "navigate": self.execute_navigation,
            "go_to": self.execute_navigation,
            "move_to": self.execute_navigation,
            "find": self.execute_perception,
            "locate": self.execute_perception,
            "grasp": self.execute_manipulation,
            "pick_up": self.execute_manipulation,
            "greet": self.execute_interaction,
            "follow": self.execute_interaction
        }

        # Action patterns for fuzzy matching
        self.action_patterns = {
            "navigate": [r"go to (.+)", r"move to (.+)", r"navigate to (.+)"],
            "grasp": [r"pick up (.+)", r"grasp (.+)", r"grab (.+)"],
            "greet": [r"greet (.+)", r"say hello to (.+)"]
        }

    def map_and_execute(self, plan_step, robot_interface):
        """Map plan step to robot action and execute"""
        action = plan_step.get("action", "").lower()
        target = plan_step.get("target", "")
        location = plan_step.get("location", "")

        if action in self.action_mapping:
            return self.action_mapping[action](target, location, robot_interface)
        else:
            # Try pattern matching
            for action_type, patterns in self.action_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, action + " " + target, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        actual_target = groups[0] if groups else target
                        return self.action_mapping[action_type](actual_target, location, robot_interface)

        self.get_logger().error(f'Unknown action: {action}')
        return False

    def execute_navigation(self, target, location, robot_interface):
        """Execute navigation action"""
        if location in robot_interface.location_map:
            x, y, theta = robot_interface.location_map[location]
            return robot_interface.navigate_to_pose(x, y, theta)
        else:
            # Try to interpret target as location
            if target in robot_interface.location_map:
                x, y, theta = robot_interface.location_map[target]
                return robot_interface.navigate_to_pose(x, y, theta)
            else:
                self.get_logger().error(f'Unknown location: {location or target}')
                return False

    def execute_perception(self, target, location, robot_interface):
        """Execute perception action"""
        # Navigate to location if specified
        if location and location in robot_interface.location_map:
            x, y, theta = robot_interface.location_map[location]
            if not robot_interface.navigate_to_pose(x, y, theta):
                return False

        # Search for target object
        return robot_interface.find_object(target)

    def execute_manipulation(self, target, location, robot_interface):
        """Execute manipulation action"""
        # Find the object first
        if not robot_interface.find_object(target):
            self.get_logger().error(f'Could not find object: {target}')
            return False

        # Execute grasp
        return robot_interface.grasp_object(target)

    def execute_interaction(self, target, location, robot_interface):
        """Execute interaction action"""
        # Find the person
        if not robot_interface.find_person(target):
            self.get_logger().error(f'Could not find person: {target}')
            return False

        # Execute interaction
        return robot_interface.interact_with_person(target)
```

### Execution Monitoring and Feedback

```python
class ExecutionMonitor:
    def __init__(self):
        self.execution_history = []
        self.max_history = 50
        self.timeout_duration = 30.0  # seconds

    def execute_plan_with_monitoring(self, plan, action_mapper, robot_interface):
        """Execute plan with monitoring and error handling"""
        for i, step in enumerate(plan):
            self.get_logger().info(f'Executing step {i+1}/{len(plan)}: {step.get("description", "")}')

            # Publish status
            status_msg = String()
            status_msg.data = f"Executing: {step.get('description', '')}"
            robot_interface.task_status_pub.publish(status_msg)

            # Execute the step
            start_time = self.get_clock().now()
            success = action_mapper.map_and_execute(step, robot_interface)

            # Check for timeout
            while not success and (self.get_clock().now() - start_time).nanoseconds / 1e9 < self.timeout_duration:
                # Retry or handle partial success
                if hasattr(robot_interface, 'check_execution_status'):
                    success = robot_interface.check_execution_status(step)
                self.get_clock().sleep_for(Duration(seconds=0.5))

            if success:
                self.get_logger().info(f'Step {i+1} completed successfully')
                self.execution_history.append({
                    'step': step,
                    'status': 'success',
                    'timestamp': self.get_clock().now().nanoseconds
                })
            else:
                self.get_logger().error(f'Step {i+1} failed')
                self.execution_history.append({
                    'step': step,
                    'status': 'failed',
                    'timestamp': self.get_clock().now().nanoseconds,
                    'error': 'Execution failed'
                })

                # Handle failure - retry, skip, or abort
                if self.should_retry(step, len(self.execution_history)):
                    self.get_logger().info('Retrying failed step...')
                    success = action_mapper.map_and_execute(step, robot_interface)
                    if success:
                        self.execution_history[-1]['status'] = 'success_after_retry'
                    else:
                        if self.should_abort_plan():
                            self.get_logger().error('Aborting plan due to failures')
                            return False
                else:
                    if self.should_skip_and_continue():
                        self.get_logger().info('Skipping failed step and continuing')
                        continue
                    else:
                        self.get_logger().error('Aborting plan')
                        return False

        # Plan completed successfully
        status_msg = String()
        status_msg.data = "Plan completed successfully"
        robot_interface.task_status_pub.publish(status_msg)
        return True

    def should_retry(self, step, failure_count):
        """Determine if a failed step should be retried"""
        # Simple retry logic - retry up to 3 times for navigation
        if step.get('action') in ['navigate', 'go_to', 'move_to'] and failure_count <= 3:
            return True
        return False

    def should_abort_plan(self):
        """Determine if the entire plan should be aborted"""
        # Abort if more than 30% of steps have failed
        total_steps = len(self.execution_history)
        failed_steps = sum(1 for h in self.execution_history if h['status'] == 'failed')
        return total_steps > 0 and (failed_steps / total_steps) > 0.3

    def should_skip_and_continue(self):
        """Determine if a failed step should be skipped"""
        # Skip if it's a non-critical action
        return True
```

## Advanced Planning Techniques

### Multi-Modal Planning

```python
class MultiModalPlanner:
    def __init__(self):
        self.vision_client = None  # Initialize vision system client
        self.speech_client = None  # Initialize speech system client

    def plan_with_multimodal_input(self, task_description, visual_context, speech_context):
        """Plan using both visual and speech context"""
        prompt = f"""
        Plan the following task considering both visual and speech context:

        Task: {task_description}

        Visual Context: {visual_context}
        Speech Context: {speech_context}

        The robot has access to:
        - Real-time camera feed
        - Object detection capabilities
        - Speech recognition
        - Navigation system
        - Manipulation capabilities

        Generate a plan that takes advantage of the visual information to adapt the task execution.
        For example, if the visual context shows obstacles, plan navigation around them.
        If speech context indicates urgency, prioritize speed over precision.

        Return the plan as JSON with spatial reasoning and adaptive actions.
        """

        # This would call the LLM with multimodal capabilities
        # In practice, you might use GPT-4V or similar multimodal models
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",  # or appropriate multimodal model
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.choices[0].message.content)
```

### Learning from Execution

```python
class AdaptivePlanner:
    def __init__(self):
        self.execution_memory = []
        self.performance_metrics = {}

    def update_plan_with_experience(self, new_task, previous_executions):
        """Update planning based on previous execution experiences"""
        if not previous_executions:
            return self.generate_new_plan(new_task)

        # Analyze previous executions for similar tasks
        similar_executions = self.find_similar_executions(new_task, previous_executions)

        if similar_executions:
            # Learn from successful patterns
            successful_patterns = [
                exec for exec in similar_executions
                if exec['outcome'] == 'success'
            ]

            if successful_patterns:
                # Generate plan biased toward successful patterns
                return self.generate_plan_with_precedent(
                    new_task, successful_patterns
                )

        # Fall back to standard planning
        return self.generate_new_plan(new_task)

    def find_similar_executions(self, new_task, previous_executions):
        """Find similar task executions using semantic similarity"""
        # This would typically use embeddings or semantic search
        # For simplicity, using keyword matching
        new_task_lower = new_task.lower()
        similar = []

        for execution in previous_executions:
            task_desc = execution.get('task_description', '').lower()
            if any(keyword in task_desc for keyword in new_task_lower.split()):
                similar.append(execution)

        return similar

    def generate_plan_with_precedent(self, task, successful_examples):
        """Generate plan based on successful precedents"""
        examples_text = "\n".join([
            f"Task: {ex['task_description']}\nPlan: {json.dumps(ex['plan'])}"
            for ex in successful_examples[:3]  # Use top 3 examples
        ])

        prompt = f"""
        Generate a plan for: {task}

        Here are successful plans for similar tasks:
        {examples_text}

        Use the successful patterns while adapting to the specific requirements of the new task.
        Return the plan as JSON.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.choices[0].message.content)
```

## Integration with ROS 2 Systems

### Planning Service

```python
from rclpy.service import Service
from std_srvs.srv import Trigger
from your_msgs.srv import PlanTask

class PlanningServiceNode(Node):
    def __init__(self):
        super().__init__('planning_service')

        # Service for task planning
        self.plan_service = self.create_service(
            PlanTask,
            'plan_task',
            self.plan_task_callback
        )

        # Initialize planner components
        self.llm_planner = LLMTaskPlannerNode()
        self.action_mapper = ActionMapper()
        self.execution_monitor = ExecutionMonitor()

    def plan_task_callback(self, request, response):
        """Service callback for task planning"""
        task_description = request.task_description
        self.get_logger().info(f'Planning task: {task_description}')

        try:
            # Generate plan using LLM
            plan = self.llm_planner.plan_task_with_llm(task_description)

            if plan:
                response.success = True
                response.plan = json.dumps(plan)
                response.message = "Plan generated successfully"
            else:
                response.success = False
                response.message = "Failed to generate plan"

        except Exception as e:
            response.success = False
            response.message = f"Planning error: {str(e)}"

        return response

# Service message definition (in your_msgs/srv/PlanTask.srv):
# string task_description
# ---
# bool success
# string plan
# string message
```

### Behavior Trees Integration

LLM planning can be integrated with behavior trees for robust execution:

```python
# Example behavior tree for LLM-guided task execution
"""
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="LLMTaskExecution">
      <RequestLLMPlan task="{task}" plan="{plan}" />
      <ExecutePlanWithRecovery plan="{plan}" />
      <ReportExecutionResult plan="{plan}" result="{result}" />
    </Sequence>
  </BehaviorTree>
</root>
"""
```

## Performance Considerations

### Caching and Optimization

```python
import functools
from typing import Dict, Any

class OptimizedLLMPlanner:
    def __init__(self):
        self.plan_cache = {}
        self.max_cache_size = 100

    @functools.lru_cache(maxsize=50)
    def get_cached_plan(self, task_hash: str) -> Dict[str, Any]:
        """Get cached plan for a task hash"""
        # This is called by the decorated method below
        pass

    def plan_task_with_caching(self, task_description: str):
        """Plan task with caching to improve performance"""
        # Create a hash of the task description for caching
        task_hash = hash(task_description.lower().strip())

        # Check cache first
        if task_hash in self.plan_cache:
            self.get_logger().info('Retrieved plan from cache')
            return self.plan_cache[task_hash]

        # Generate new plan
        plan = self.plan_task_with_llm(task_description)

        # Cache the result
        if len(self.plan_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.plan_cache))
            del self.plan_cache[oldest_key]

        self.plan_cache[task_hash] = plan
        return plan

    def plan_with_template_matching(self, task_description: str):
        """Use template matching for common tasks before LLM"""
        # Define common task templates
        templates = {
            r"bring me (a|the|some) (.+)": "fetch_object",
            r"go to the (.+)": "navigate_to_location",
            r"find (a|the) (.+)": "search_for_object",
            r"clean the (.+)": "clean_area"
        }

        for pattern, template_type in templates.items():
            match = re.search(pattern, task_description, re.IGNORECASE)
            if match:
                # Use template-specific planning (faster than LLM)
                return self.plan_with_template(template_type, match.groups())

        # Fall back to LLM planning for complex tasks
        return self.plan_task_with_llm(task_description)

    def plan_with_template(self, template_type: str, params: tuple):
        """Plan using predefined templates"""
        if template_type == "fetch_object":
            object_name = params[-1]  # Last captured group
            return [
                {"action": "find", "target": object_name, "description": f"Find the {object_name}"},
                {"action": "navigate", "target": object_name, "description": f"Go to the {object_name}"},
                {"action": "grasp", "target": object_name, "description": f"Pick up the {object_name}"},
                {"action": "navigate", "target": "delivery_location", "description": "Return to delivery location"}
            ]
        # Add more templates as needed
        return []
```

## Safety and Validation

### Plan Validation

```python
class PlanValidator:
    def __init__(self):
        self.safety_constraints = [
            "don't enter restricted areas",
            "don't interact with dangerous objects",
            "don't leave designated operational zone",
            "don't perform actions that could harm humans"
        ]

    def validate_plan(self, plan, environment_context):
        """Validate plan against safety constraints"""
        validation_prompt = f"""
        Validate the following robot task plan against safety constraints:

        Plan: {json.dumps(plan, indent=2)}

        Environment context: {environment_context}

        Safety constraints: {self.safety_constraints}

        Return validation result as JSON with:
        - "is_safe": boolean
        - "issues": list of safety issues found
        - "suggestions": list of modifications to make plan safe
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": validation_prompt}]
        )

        validation_result = json.loads(response.choices[0].message.content)

        if not validation_result.get("is_safe", True):
            self.get_logger().warn(f'Safety issues found: {validation_result.get("issues")}')
            return False, validation_result.get("suggestions", [])

        return True, []
```

## Best Practices

### Prompt Engineering

- **Clear structure**: Use consistent JSON output formats
- **Examples**: Provide few-shot examples for better results
- **Constraints**: Clearly specify robot capabilities and limitations
- **Context**: Include relevant environmental and situational context

### Error Handling

- **Graceful degradation**: Have fallback plans when LLM fails
- **Human in the loop**: Allow human intervention when needed
- **Partial execution**: Continue with remaining tasks when one fails
- **State recovery**: Restore robot state after failed executions

### Performance Optimization

- **Caching**: Cache common plans and patterns
- **Template matching**: Use templates for frequent tasks
- **Parallel execution**: Execute independent subtasks in parallel
- **Incremental planning**: Plan only what's needed for immediate future

## Troubleshooting

### Common Issues

- **Hallucination**: LLM generates impossible actions
- **Context overflow**: Too much information for LLM to process
- **Execution drift**: Real-world differs from LLM's assumptions
- **Performance**: LLM calls taking too long for real-time operation

## Summary

LLM-based task planning enables sophisticated robotic behaviors by translating high-level human instructions into executable robot actions. Success requires careful prompt engineering, robust execution monitoring, and proper integration with robotic systems. The combination of LLM reasoning capabilities with real-world robot execution creates powerful autonomous systems capable of complex task completion.