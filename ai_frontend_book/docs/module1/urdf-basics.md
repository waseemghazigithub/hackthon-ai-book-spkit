---
sidebar_position: 4
---

# Humanoid Robot Structure (URDF)

## Introduction

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including its links, joints, and kinematic structure. For humanoid robots, URDF is essential for simulation, visualization, and control.

## Core Components

### Links

A link represents a rigid body in the robot. Each link has:
- **Visual properties**: How the link appears (shape, color, material)
- **Collision properties**: How the link interacts with other objects in simulation
- **Inertial properties**: Mass, center of mass, and inertia tensor for physics simulation

```xml
<link name="base_link">
  <visual>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 0.8 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.6" radius="0.2"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

### Joints

Joints connect links and define their relative motion. The main joint types are:
- **Fixed**: No motion between links
- **Revolute**: Rotational motion around an axis
- **Continuous**: Like revolute but unlimited rotation
- **Prismatic**: Linear motion along an axis
- **Floating**: 6-DOF motion
- **Planar**: Motion on a plane

```xml
<joint name="base_to_wheel" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0.1 -0.2 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>
```

### Frames

URDF defines coordinate frames for each link. These frames are essential for:
- Kinematic calculations
- Sensor data interpretation
- Robot control
- Visualization

## Humanoid-Specific Considerations

### Humanoid Joint Structure

Humanoid robots typically have a structure similar to the human body:

```
base_link (torso)
├── head_link
├── left_upper_arm
│   ├── left_lower_arm
│   └── left_hand
├── right_upper_arm
│   ├── right_lower_arm
│   └── right_hand
├── left_upper_leg
│   ├── left_lower_leg
│   └── left_foot
└── right_upper_leg
    ├── right_lower_leg
    └── right_foot
```

### Kinematic Chains

Humanoid robots have multiple kinematic chains:
- **Arms**: For manipulation tasks
- **Legs**: For locomotion and balance
- **Spine**: For posture and movement
- **Neck**: For head orientation

## URDF in ROS 2 and Simulators

### Integration with ROS 2

URDF models are used in ROS 2 for:
- Robot state publishing (robot_state_publisher)
- TF (Transform) tree generation
- Simulation in Gazebo/Isaac Sim
- Motion planning with MoveIt
- Visualization in RViz

### Robot State Publisher

The robot_state_publisher node takes the URDF and joint positions to publish the TF tree:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)
```

## Creating Humanoid URDF Models

### Basic Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

### Best Practices

1. **Use consistent naming**: Follow conventions for link and joint names
2. **Define proper inertial properties**: Essential for accurate simulation
3. **Use appropriate collision geometry**: Simplified shapes for performance
4. **Set realistic joint limits**: Based on physical constraints
5. **Validate the model**: Check for self-collisions and proper kinematics

## Reading and Reasoning about URDF

### Understanding Robot Kinematics

When reading a URDF file:
1. Identify the base link (usually torso for humanoid robots)
2. Trace the kinematic chains from parent to child links
3. Understand the degrees of freedom provided by each joint
4. Note the coordinate frame orientations

### Common URDF Patterns

- **Chain structure**: Links connected in series (like arms/legs)
- **Tree structure**: Multiple branches from a central link (like torso with arms and legs)
- **Fixed connections**: Links with no relative motion
- **Actuated joints**: Joints with controllers in the real robot

## Tools for Working with URDF

### Visualization
- **RViz**: Real-time visualization of robot models
- **Gazebo/Isaac Sim**: Physics simulation
- **Blender**: 3D modeling and editing

### Validation
- **check_urdf**: Command-line tool to check URDF syntax
- **urdf_tutorial**: Examples and best practices

## Summary

URDF is fundamental to humanoid robotics in ROS 2, providing the robot description needed for simulation, visualization, and control. Understanding links, joints, and frames is crucial for working with humanoid robot models. Proper URDF models enable accurate simulation and effective robot control in both virtual and physical environments.