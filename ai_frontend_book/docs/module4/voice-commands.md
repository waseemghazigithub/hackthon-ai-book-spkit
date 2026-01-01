---
sidebar_position: 2
---

# Voice Commands with Whisper

## Introduction

Voice commands enable natural human-robot interaction by allowing users to communicate with robots using spoken language. This chapter focuses on implementing voice command processing using OpenAI's Whisper model, which provides state-of-the-art speech recognition capabilities for converting spoken commands into text that can be processed by AI systems.

## Speech Recognition Pipeline

### Audio Capture and Preprocessing

The voice command system starts with capturing audio from the environment:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import wave
import threading

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Audio capture parameters
        self.rate = 16000  # 16kHz sampling rate
        self.chunk = 1024  # Audio chunk size
        self.format = pyaudio.paInt16
        self.channels = 1  # Mono audio
        self.record_seconds = 5  # Maximum recording time

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Publisher for recognized commands
        self.command_pub = self.create_publisher(String, 'voice_command', 10)

        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.capture_thread.start()

    def capture_audio(self):
        """Continuously capture audio and process voice commands"""
        while rclpy.ok():
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            # Listen for voice activity
            frames = []
            silence_threshold = 500  # Adjust based on environment
            silent_chunks = 0
            max_silent_chunks = 30  # Stop recording after 30 silent chunks

            # Start recording when voice activity is detected
            recording = False
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Check for voice activity
                if np.max(np.abs(audio_data)) > silence_threshold:
                    recording = True
                    frames.append(data)
                    silent_chunks = 0
                elif recording:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        break
                elif not recording:
                    # Continue listening for voice activity
                    continue

            # Close audio stream
            stream.stop_stream()
            stream.close()

            # Process recorded audio if we captured something
            if len(frames) > 0 and recording:
                self.process_audio(frames)

    def process_audio(self, frames):
        """Process recorded audio with Whisper"""
        # Convert frames to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Process with Whisper (in practice, this would call the Whisper model)
        command_text = self.transcribe_with_whisper(audio_array)

        if command_text:
            # Publish recognized command
            cmd_msg = String()
            cmd_msg.data = command_text.strip()
            self.command_pub.publish(cmd_msg)
            self.get_logger().info(f'Recognized command: {command_text}')

    def transcribe_with_whisper(self, audio_array):
        """Transcribe audio using Whisper model"""
        # In practice, this would interface with the Whisper model
        # For now, we'll simulate the transcription
        # This could be done using openai's API or local Whisper model
        import whisper

        # Convert audio to the format expected by Whisper
        # Whisper expects audio at 16kHz sample rate
        # Process with local Whisper model
        model = whisper.load_model("base")  # Load appropriate model
        result = model.transcribe(audio_array)
        return result["text"]
```

### Whisper Model Integration

Whisper can be used in two ways: via OpenAI API or as a local model:

```python
import openai
import whisper
import torch

class WhisperProcessor:
    def __init__(self, use_api=True):
        self.use_api = use_api
        if not use_api:
            # Load local Whisper model
            self.model = whisper.load_model("base")
        else:
            # Set up OpenAI API key
            openai.api_key = "your-api-key-here"

    def transcribe(self, audio_data, language="en"):
        """Transcribe audio to text using Whisper"""
        if self.use_api:
            return self.transcribe_with_api(audio_data)
        else:
            return self.transcribe_locally(audio_data, language)

    def transcribe_with_api(self, audio_data):
        """Transcribe using OpenAI's API"""
        # Save audio to temporary file for API call
        temp_filename = "temp_audio.wav"
        self.save_audio_file(audio_data, temp_filename)

        with open(temp_filename, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
            return response["text"]

    def transcribe_locally(self, audio_data, language):
        """Transcribe using local Whisper model"""
        result = self.model.transcribe(audio_data, language=language)
        return result["text"]

    def save_audio_file(self, audio_data, filename):
        """Save audio data to WAV file"""
        import wave
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data.tobytes())
```

## Command Processing and Intent Recognition

### Natural Language Understanding

After transcription, the text needs to be processed to extract the intent:

```python
import re
from enum import Enum

class RobotCommand(Enum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    GOTO_LOCATION = "goto_location"
    GRAB_OBJECT = "grab_object"
    FOLLOW_ME = "follow_me"
    TAKE_PICTURE = "take_picture"

class CommandProcessor:
    def __init__(self):
        # Define command patterns
        self.command_patterns = {
            RobotCommand.MOVE_FORWARD: [
                r"move forward",
                r"go forward",
                r"move ahead",
                r"go ahead",
                r"forward"
            ],
            RobotCommand.MOVE_BACKWARD: [
                r"move backward",
                r"go backward",
                r"move back",
                r"go back",
                r"backward",
                r"back"
            ],
            RobotCommand.TURN_LEFT: [
                r"turn left",
                r"rotate left",
                r"turn to the left",
                r"pivot left"
            ],
            RobotCommand.TURN_RIGHT: [
                r"turn right",
                r"rotate right",
                r"turn to the right",
                r"pivot right"
            ],
            RobotCommand.STOP: [
                r"stop",
                r"halt",
                r"pause",
                r"freeze"
            ],
            RobotCommand.GOTO_LOCATION: [
                r"go to (.+)",
                r"move to (.+)",
                r"navigate to (.+)",
                r"go to the (.+)"
            ],
            RobotCommand.FOLLOW_ME: [
                r"follow me",
                r"follow",
                r"come with me",
                r"follow me around"
            ],
            RobotCommand.TAKE_PICTURE: [
                r"take picture",
                r"take photo",
                r"capture image",
                r"take a picture",
                r"take a photo"
            ]
        }

    def process_command(self, text):
        """Process text command and extract intent and parameters"""
        text = text.lower().strip()

        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract parameters if any
                    if match.groups():
                        return command_type, match.groups()
                    else:
                        return command_type, []

        # If no pattern matches, return None
        return None, []

    def validate_command(self, command_type, params):
        """Validate that the command is appropriate"""
        if command_type == RobotCommand.GOTO_LOCATION and params:
            # Validate location parameter
            location = params[0]
            valid_locations = ["kitchen", "living room", "bedroom", "office", "dining room"]
            return location in valid_locations

        return True
```

### Voice Command Node with Processing

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import threading

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')

        # Subscribe to voice commands
        self.command_sub = self.create_subscription(
            String, 'voice_command', self.command_callback, 10)

        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize command processor
        self.command_processor = CommandProcessor()

        # Location map for navigation
        self.location_map = {
            "kitchen": (3.0, 2.0, 0.0),  # x, y, theta
            "living room": (0.0, 0.0, 0.0),
            "bedroom": (-2.0, 3.0, 1.57),
            "office": (4.0, -1.0, -1.57),
            "dining room": (1.0, -2.0, 0.0)
        }

    def command_callback(self, msg):
        """Process incoming voice command"""
        command_text = msg.data
        self.get_logger().info(f'Processing command: {command_text}')

        # Extract intent and parameters
        command_type, params = self.command_processor.process_command(command_text)

        if command_type:
            if self.command_processor.validate_command(command_type, params):
                self.execute_command(command_type, params)
            else:
                self.get_logger().warn(f'Invalid command: {command_text}')
        else:
            self.get_logger().warn(f'Unrecognized command: {command_text}')

    def execute_command(self, command_type, params):
        """Execute the recognized command"""
        if command_type == RobotCommand.MOVE_FORWARD:
            self.move_robot(0.2, 0.0)  # Move forward at 0.2 m/s
        elif command_type == RobotCommand.MOVE_BACKWARD:
            self.move_robot(-0.2, 0.0)  # Move backward at 0.2 m/s
        elif command_type == RobotCommand.TURN_LEFT:
            self.move_robot(0.0, 0.5)  # Turn left at 0.5 rad/s
        elif command_type == RobotCommand.TURN_RIGHT:
            self.move_robot(0.0, -0.5)  # Turn right at 0.5 rad/s
        elif command_type == RobotCommand.STOP:
            self.stop_robot()
        elif command_type == RobotCommand.GOTO_LOCATION and params:
            location = params[0]
            if location in self.location_map:
                x, y, theta = self.location_map[location]
                self.navigate_to_pose(x, y, theta)
            else:
                self.get_logger().warn(f'Unknown location: {location}')
        elif command_type == RobotCommand.FOLLOW_ME:
            self.start_follow_mode()
        elif command_type == RobotCommand.TAKE_PICTURE:
            self.take_picture()
        else:
            self.get_logger().warn(f'Command not implemented: {command_type}')

    def move_robot(self, linear_vel, angular_vel):
        """Send velocity commands to robot"""
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        """Stop robot movement"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def navigate_to_pose(self, x, y, theta):
        """Navigate to specified pose using Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        import math
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Send navigation goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_done_callback)

    def navigation_done_callback(self, future):
        """Handle navigation completion"""
        goal_handle = future.result()
        if goal_handle.accepted:
            self.get_logger().info('Navigation goal accepted')
        else:
            self.get_logger().error('Navigation goal rejected')

    def start_follow_mode(self):
        """Start person following mode"""
        # This would typically activate a person following node
        self.get_logger().info('Starting person following mode')

    def take_picture(self):
        """Take a picture using robot's camera"""
        # This would typically trigger camera capture
        self.get_logger().info('Taking picture')
```

## Performance Optimization

### Real-time Processing

For real-time voice command processing:

```python
import queue
import threading
from collections import deque

class RealTimeVoiceProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Audio processing parameters
        self.sample_rate = 16000
        self.buffer_duration = 1.0  # 1 second buffer
        self.buffer_size = int(self.sample_rate * self.buffer_duration)

        # Circular buffer for continuous audio
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # Start processing threads
        self.processing_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        self.processing_thread.start()

    def add_audio_chunk(self, audio_data):
        """Add audio chunk to processing queue"""
        self.audio_queue.put(audio_data)

    def process_audio_stream(self):
        """Continuously process audio stream"""
        while True:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Add to circular buffer
                self.audio_buffer.extend(audio_chunk)

                # Check for voice activity
                if self.detect_voice_activity():
                    # Process the buffered audio
                    audio_segment = list(self.audio_buffer)
                    command = self.process_segment(audio_segment)

                    if command:
                        self.command_queue.put(command)

            except queue.Empty:
                continue  # Continue waiting for audio

    def detect_voice_activity(self):
        """Detect voice activity in audio buffer"""
        if len(self.audio_buffer) < 1000:  # Need minimum buffer size
            return False

        # Convert to numpy array for processing
        audio_array = np.array(list(self.audio_buffer)[-1000:])  # Last 1000 samples

        # Calculate energy-based voice activity detection
        energy = np.mean(np.abs(audio_array) ** 2)
        threshold = 0.001  # Adjust based on environment

        return energy > threshold

    def process_segment(self, audio_segment):
        """Process audio segment with Whisper"""
        # Convert to appropriate format for Whisper
        audio_array = np.array(audio_segment).astype(np.float32) / 32768.0

        # Process with Whisper (simplified)
        # In practice, you'd use the Whisper model here
        return self.transcribe_audio(audio_array)

    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper model"""
        # Placeholder for Whisper transcription
        # This would call the actual Whisper model
        return "command text"
```

## Integration with Robotics Systems

### ROS 2 Integration Patterns

The voice command system integrates with ROS 2 through standard message passing:

```yaml
# Launch file for voice command system
voice_command_launch.py:
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_command_pkg',
            executable='audio_capture_node',
            name='audio_capture',
            parameters=[
                {'sample_rate': 16000},
                {'channels': 1},
                {'chunk_size': 1024}
            ]
        ),
        Node(
            package='voice_command_pkg',
            executable='whisper_processor_node',
            name='whisper_processor',
            parameters=[
                {'model_size': 'base'},
                {'use_local_model': True}
            ]
        ),
        Node(
            package='voice_command_pkg',
            executable='command_interpreter_node',
            name='command_interpreter',
            parameters=[
                {'command_timeout': 5.0}
            ]
        )
    ])
```

## Error Handling and Robustness

### Handling Recognition Errors

```python
class RobustVoiceCommandProcessor:
    def __init__(self):
        self.command_history = []
        self.max_history = 10
        self.confidence_threshold = 0.7  # Minimum confidence for command acceptance

    def process_command_with_confidence(self, text, confidence):
        """Process command with confidence scoring"""
        if confidence < self.confidence_threshold:
            self.get_logger().warn(f'Low confidence command: {text} (confidence: {confidence})')
            # Could ask for confirmation or ignore
            return False

        command_type, params = self.command_processor.process_command(text)

        if command_type:
            self.command_history.append((command_type, params, confidence))
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)

            self.execute_command(command_type, params)
            return True
        else:
            return False

    def handle_recognition_error(self, error_type):
        """Handle different types of recognition errors"""
        if error_type == "no_speech_detected":
            self.get_logger().info("No speech detected, continuing to listen")
        elif error_type == "audio_too_noisy":
            self.get_logger().warn("Audio too noisy, consider adjusting microphone or environment")
        elif error_type == "recognition_failed":
            self.get_logger().error("Speech recognition failed")
```

## Best Practices

### Audio Quality Considerations

- **Microphone placement**: Position microphone for optimal voice capture
- **Noise reduction**: Use noise cancellation techniques
- **Audio preprocessing**: Apply filters to improve signal quality
- **Sampling rate**: Use 16kHz for optimal Whisper performance

### Performance Optimization

- **Model selection**: Choose appropriate Whisper model size for your hardware
- **Batch processing**: Process audio in chunks for efficiency
- **Caching**: Cache frequently recognized phrases
- **Fallback mechanisms**: Provide alternative input methods

### Privacy and Security

- **Local processing**: Consider processing sensitive commands locally
- **Data retention**: Limit storage of audio data
- **Encryption**: Encrypt audio data in transit
- **Access controls**: Restrict access to voice command systems

## Troubleshooting

### Common Issues

- **Poor recognition**: Check audio quality and background noise
- **High latency**: Optimize model size or processing pipeline
- **False positives**: Adjust voice activity detection thresholds
- **Resource usage**: Monitor CPU/memory usage during processing

## Summary

Voice commands provide a natural interface for human-robot interaction, enabling users to control robots through spoken language. Implementing Whisper-based voice recognition requires careful attention to audio quality, real-time processing, and command interpretation. The integration with ROS 2 systems enables voice commands to control robot behavior, navigation, and other capabilities.