---
sidebar_position: 3
---

# Python AI Agents with ROS 2 (rclpy)

## Introduction

Python AI agents in ROS 2 are implemented using rclpy, the Python client library for ROS 2. This allows AI developers to create intelligent agents that can communicate with robot systems using ROS 2's messaging infrastructure.

## Creating Python ROS 2 Nodes

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class AIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent_node')
        # Node initialization code here
        self.get_logger().info('AI Agent Node initialized')

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AIAgentNode()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle

Python ROS 2 nodes follow a specific lifecycle:
1. **Initialization**: Node is created and registered with the ROS 2 system
2. **Execution**: Node performs its tasks, handling messages and callbacks
3. **Shutdown**: Node cleans up resources and unregisters from the system

## Publishing Messages

Publishing allows a node to send data to other nodes through topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class AIPublisherNode(Node):
    def __init__(self):
        super().__init__('ai_publisher')
        self.publisher = self.create_publisher(String, 'ai_decisions', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'AI Decision: Processed data point {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

## Subscribing to Messages

Subscribing allows a node to receive data from other nodes:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class AISubscriberNode(Node):
    def __init__(self):
        super().__init__('ai_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received sensor data: "{msg.data}"')
        # Process the received data with AI logic
        decision = self.make_ai_decision(msg.data)
        self.get_logger().info(f'AI Decision: {decision}')

    def make_ai_decision(self, sensor_data):
        # Simple AI decision logic
        return f"Decision based on: {sensor_data}"
```

## Service Calls

Services provide synchronous request/response communication:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AIClientNode(Node):
    def __init__(self):
        super().__init__('ai_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.client.call_async(request)
        return future
```

## The AI Agent → Robot Controller Flow

### Data Processing Pipeline

1. **Sensor Data Reception**: AI agent subscribes to sensor topics
2. **AI Processing**: Raw sensor data is processed through AI algorithms
3. **Decision Making**: AI agent determines appropriate actions
4. **Command Transmission**: Decisions are sent to robot controllers
5. **Feedback Loop**: Results from robot execution are received and processed

### Example Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class NavigationAIAgent(Node):
    def __init__(self):
        super().__init__('navigation_ai_agent')

        # Subscribe to sensor data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

        # Publish commands to robot
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.obstacle_threshold = 1.0  # meters

    def scan_callback(self, msg):
        # Process sensor data
        min_distance = min(msg.ranges)

        # AI decision making
        cmd = Twist()
        if min_distance < self.obstacle_threshold:
            # Obstacle detected - turn
            cmd.angular.z = 0.5
        else:
            # Clear path - move forward
            cmd.linear.x = 0.2

        # Send command to robot
        self.publisher.publish(cmd)
```

## Best Practices

### Error Handling

Always implement proper error handling in your AI agents:

```python
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    node.get_logger().info('Node interrupted by user')
except Exception as e:
    node.get_logger().error(f'Error in node: {e}')
finally:
    node.destroy_node()
    rclpy.shutdown()
```

### Resource Management

Properly clean up resources when the node shuts down:

- Destroy publishers, subscribers, and clients
- Cancel timers
- Close any external connections

### Performance Considerations

- Use appropriate QoS (Quality of Service) settings for your application
- Consider message frequency and processing time
- Implement proper threading if needed for complex AI processing

## Summary

Python AI agents using rclpy form the bridge between AI algorithms and robot control systems. By implementing publishers, subscribers, and service clients, AI agents can receive sensor data, process it through intelligent algorithms, and send commands to robot controllers. This creates the flow from AI agent to robot controller that is essential for physical AI applications.