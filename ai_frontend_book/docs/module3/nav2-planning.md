---
sidebar_position: 4
---

# Nav2 Path Planning

## Introduction

Navigation2 (Nav2) is the ROS 2 navigation stack that provides path planning, navigation, and obstacle avoidance capabilities for mobile robots. It's the successor to ROS 1's navigation stack, designed with modern robotics requirements in mind, including support for dynamic environments, multiple robot systems, and advanced path planning algorithms.

## Architecture Overview

### Core Components

Nav2 consists of several key components working together:

- **Navigation Server**: Main orchestrator of the navigation system
- **Planner Server**: Handles global path planning
- **Controller Server**: Manages local path following and obstacle avoidance
- **Recovery Server**: Provides behavior trees for recovery actions
- **Lifecycle Manager**: Manages the lifecycle of navigation components

### Execution Flow

The navigation process follows this sequence:
1. **Goal Reception**: Navigation server receives a goal pose
2. **Global Planning**: Planner server computes a global path
3. **Local Planning**: Controller server follows the path while avoiding obstacles
4. **Execution Monitoring**: System monitors progress and handles failures
5. **Recovery**: Recovery server executes recovery behaviors if needed

## Global Path Planning

### Costmap Integration

Global planners use costmaps to understand the environment:

```yaml
# Global costmap configuration
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        inflation_radius: 0.55
```

### Available Planners

Nav2 supports multiple global planners:

- **NavFn**: Fast marching method for path planning
- **Global Planner**: A* implementation
- **Theta*: Theta* implementation for any-angle planning
- **SMAC Planner**: Sampling-based motion planning for SE2 state space

### Custom Planner Implementation

```python
import rclpy
from rclpy.node import Node
from nav2_core import GlobalPlanner
from nav2_costmap_2d import Costmap2D
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np

class CustomGlobalPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__('custom_global_planner')
        self.costmap = None
        self.tf_buffer = None

    def configure(self, tf_buffer, costmap_ros, lifecycle_node, planner_name):
        self.costmap = costmap_ros.get_costmap()
        self.tf_buffer = tf_buffer
        self.planner_node = lifecycle_node

    def cleanup(self):
        pass

    def set_costmap(self, costmap):
        self.costmap = costmap

    def create_plan(self, start, goal):
        # Convert poses to costmap coordinates
        start_x = int((start.pose.position.x - self.costmap.getOriginX()) / self.costmap.getResolution())
        start_y = int((start.pose.position.y - self.costmap.getOriginY()) / self.costmap.getResolution())

        goal_x = int((goal.pose.position.x - self.costmap.getOriginX()) / self.costmap.getResolution())
        goal_y = int((goal.pose.position.y - self.costmap.getOriginY()) / self.costmap.getResolution())

        # Plan path using custom algorithm
        path_points = self.plan_path_astar(start_x, start_y, goal_x, goal_y)

        # Convert path back to world coordinates
        path = Path()
        path.header.frame_id = "map"

        for point in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = point[0] * self.costmap.getResolution() + self.costmap.getOriginX()
            pose.pose.position.y = point[1] * self.costmap.getResolution() + self.costmap.getOriginY()
            pose.pose.position.z = 0.0
            path.poses.append(pose)

        return path

    def plan_path_astar(self, start_x, start_y, goal_x, goal_y):
        # A* path planning algorithm implementation
        # This is a simplified version
        open_set = [(start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, goal_x, goal_y)}

        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == (goal_x, goal_y):
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)

            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if (0 <= neighbor[0] < self.costmap.getSizeInCellsX() and
                    0 <= neighbor[1] < self.costmap.getSizeInCellsY()):

                    # Check if cell is free (not occupied)
                    cost = self.costmap.getCost(neighbor[0], neighbor[1])
                    if cost >= 253:  # Considered occupied
                        continue

                    tentative_g_score = g_score[current] + 1  # Simplified cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor[0], neighbor[1], goal_x, goal_y)

                        if neighbor not in open_set:
                            open_set.append(neighbor)

        return []  # No path found

    def heuristic(self, x1, y1, x2, y2):
        # Manhattan distance heuristic
        return abs(x1 - x2) + abs(y1 - y2)
```

## Local Path Following

### Controller Server

The controller server manages local path following:

```yaml
# Controller server configuration
controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
      # Inner controller
      inner_controller_plugin: "FollowPath"
      inner_controller_topic: "inner_controller"
      # Rotation shim parameters
      rotation_speed_thresh: 0.3
      min_error_to_rotate: 0.3
      time_to_allow_still: 0.0
```

### Available Controllers

- **DWB (Dynamic Window Approach)**: Local trajectory optimization
- **TEB (Timed Elastic Band)**: Time-optimal trajectory planning
- **MPC (Model Predictive Control)**: Advanced control with prediction
- **Rotation Shim Controller**: Combines rotation and path following

### Local Planner Implementation

```python
import rclpy
from rclpy.node import Node
from nav2_core import Controller
from nav2_costmap_2d import Costmap2D
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import TransformException
import math

class CustomLocalController(Controller):
    def __init__(self):
        super().__init__('custom_local_controller')
        self.costmap = None
        self.tf_buffer = None
        self.current_path = None

    def configure(self, tf_buffer, costmap_ros, lifecycle_node, controller_name):
        self.costmap = costmap_ros.get_costmap()
        self.tf_buffer = tf_buffer
        self.lifecycle_node = lifecycle_node

    def cleanup(self):
        pass

    def setPlan(self, path):
        self.current_path = path

    def computeVelocityCommands(self, pose, velocity):
        # Get current robot pose and velocity
        current_pose = pose
        current_velocity = velocity

        # Calculate desired velocity based on path
        cmd_vel = Twist()

        if self.current_path and len(self.current_path.poses) > 0:
            # Find closest point on path
            target_point = self.getClosestPointOnPath(current_pose)

            if target_point:
                # Calculate direction to target
                dx = target_point.pose.position.x - current_pose.pose.position.x
                dy = target_point.pose.position.y - current_pose.pose.position.y

                # Calculate desired heading
                desired_yaw = math.atan2(dy, dx)
                current_yaw = self.getYawFromQuaternion(current_pose.pose.orientation)

                # Calculate heading error
                heading_error = self.normalizeAngle(desired_yaw - current_yaw)

                # Simple proportional controller for rotation
                angular_velocity = min(max(heading_error * 1.0, -1.0), 1.0)

                # Calculate distance to target
                distance = math.sqrt(dx*dx + dy*dy)

                # Set linear velocity based on distance
                linear_velocity = min(distance * 0.5, 0.5)  # Max 0.5 m/s

                cmd_vel.linear.x = linear_velocity
                cmd_vel.angular.z = angular_velocity

        return cmd_vel

    def getClosestPointOnPath(self, current_pose):
        if not self.current_path or len(self.current_path.poses) == 0:
            return None

        closest_point = None
        min_distance = float('inf')

        for pose in self.current_path.poses:
            dx = pose.pose.position.x - current_pose.pose.position.x
            dy = pose.pose.position.y - current_pose.pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < min_distance:
                min_distance = distance
                closest_point = pose

        return closest_point

    def getYawFromQuaternion(self, orientation):
        # Extract yaw from quaternion
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalizeAngle(self, angle):
        # Normalize angle to [-pi, pi]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
```

## Behavior Trees for Navigation

### BT Concepts

Nav2 uses behavior trees for navigation execution:

- **Sequence**: Execute nodes in order until one fails
- **Fallback**: Try nodes until one succeeds
- **Decorator**: Modify behavior of child nodes
- **Action**: Perform specific navigation task

### Sample Behavior Tree

```xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithRecovery">
        <RateController hz="1.0" name="RateController">
          <GoalUpdater input_port="goal" name="GoalUpdater" output_port="goal">
            <RecoveryNode number_of_retries="2" name="ComputePathToPose">
              <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
              <RecoveryNode number_of_retries="1" name="SmoothPath">
                <SmoothPath input_path="{path}" output_path="{path}" smoother_id="simple_smoother"/>
              </RecoveryNode>
            </RecoveryNode>
          </GoalUpdater>
        </RateController>
        <RecoveryNode number_of_retries="4" name="FollowPath">
          <FollowPath path="{path}" controller_id="FollowPath"/>
          <Spin spin_dist="1.57"/>
        </RecoveryNode>
      </PipelineSequence>
      <RecoveryNode number_of_retries="2" name="ClearingRotation">
        <ClearEntirelyCostmap costmap_port="global" name="GlobalPlannerClear"/>
        <ClearEntirelyCostmap costmap_port="local" name="LocalPlannerClear"/>
        <Spin spin_dist="3.14"/>
      </RecoveryNode>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

## Configuration and Tuning

### Parameter Configuration

```yaml
# Main navigation configuration
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - "BackUp"
      - "Spin"
      - "Wait"
      - "ClearEntirelyCostmap"
      - "SimpleGoalChecker"
      - "TransformBackPose"
```

### Costmap Tuning

```yaml
# Costmap parameters for optimal navigation
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

## Advanced Features

### Multi-Robot Navigation

Nav2 supports multi-robot navigation with coordination:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Action clients for multiple robots
        self.robot1_client = ActionClient(self, NavigateToPose, 'robot1/navigate_to_pose')
        self.robot2_client = ActionClient(self, NavigateToPose, 'robot2/navigate_to_pose')

        # Topic for inter-robot communication
        self.collision_avoidance_pub = self.create_publisher(
            PoseStamped, 'collision_avoidance_goals', 10)

    def coordinate_robots(self, robot1_goal, robot2_goal):
        # Check for potential collisions
        if self.will_collide(robot1_goal, robot2_goal):
            # Adjust goals to avoid collision
            adjusted_goals = self.adjust_goals_for_collision(robot1_goal, robot2_goal)
            robot1_goal, robot2_goal = adjusted_goals

        # Send navigation requests
        self.send_navigation_request(self.robot1_client, robot1_goal)
        self.send_navigation_request(self.robot2_client, robot2_goal)

    def will_collide(self, goal1, goal2):
        # Check if robot paths will intersect
        # Implementation depends on path prediction
        pass

    def adjust_goals_for_collision(self, goal1, goal2):
        # Adjust goals to avoid collision
        # Could involve temporal or spatial separation
        pass
```

### Dynamic Obstacle Avoidance

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
import numpy as np

class DynamicObstacleAvoider(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_avoider')

        # Subscribe to dynamic obstacle detection
        self.obstacle_sub = self.create_subscription(
            PointCloud2, 'dynamic_obstacles', self.obstacle_callback, 10)

        # Subscribe to current velocity command
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Publish modified velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)

        self.current_cmd_vel = Twist()
        self.dynamic_obstacles = []

    def obstacle_callback(self, msg):
        # Process dynamic obstacles
        self.dynamic_obstacles = self.process_pointcloud(msg)

    def cmd_vel_callback(self, msg):
        self.current_cmd_vel = msg

    def process_pointcloud(self, msg):
        # Extract dynamic obstacle information from point cloud
        obstacles = []
        # Implementation to extract obstacle positions, velocities, etc.
        return obstacles

    def dynamic_avoidance(self):
        # Modify velocity command based on dynamic obstacles
        safe_cmd_vel = Twist()

        if self.dynamic_obstacles:
            # Calculate safe velocity based on obstacle positions
            safe_cmd_vel = self.calculate_safe_velocity(
                self.current_cmd_vel, self.dynamic_obstacles)
        else:
            safe_cmd_vel = self.current_cmd_vel

        self.cmd_vel_pub.publish(safe_cmd_vel)

    def calculate_safe_velocity(self, desired_vel, obstacles):
        # Implement dynamic window approach or similar algorithm
        # to avoid moving obstacles
        pass
```

## Best Practices

### Performance Optimization

- **Costmap Resolution**: Balance accuracy with performance
- **Update Frequencies**: Optimize for your robot's speed
- **Path Smoothing**: Apply smoothing for better execution
- **Recovery Behaviors**: Implement appropriate recovery actions

### Safety Considerations

- **Velocity Limits**: Set appropriate speed limits
- **Inflation Radius**: Configure for robot size and safety margin
- **Sensor Fusion**: Combine multiple sensor inputs
- **Emergency Stops**: Implement safety mechanisms

### Testing and Validation

- **Simulation Testing**: Test in simulation first
- **Gradual Deployment**: Start with simple scenarios
- **Parameter Tuning**: Adjust for your specific robot
- **Edge Cases**: Test boundary conditions

## Troubleshooting

### Common Issues

- **Path Planning Failures**: Check costmap configuration and map quality
- **Local Minima**: Adjust controller parameters or add recovery behaviors
- **Oscillation**: Tune controller gains and parameters
- **Performance**: Optimize costmap resolution and update rates

## Summary

Nav2 provides a comprehensive navigation solution for mobile robots in ROS 2. Its flexible architecture, behavior tree execution, and extensive configuration options make it suitable for a wide range of navigation tasks. Understanding global and local planning, behavior trees, and parameter tuning is essential for successful navigation system implementation.