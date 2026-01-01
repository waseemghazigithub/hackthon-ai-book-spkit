---
sidebar_position: 2
---

# ROS 2 Fundamentals

## Introduction

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Core Concepts

### Nodes
A node is a process that performs computation. ROS 2 is designed with a distributed architecture where individual programs that may be running on different machines are organized as nodes that make up the whole system. Nodes written in different programming languages can communicate with each other.

### Topics
Topics are named buses over which nodes exchange messages. A node can publish messages to a topic, and other nodes can subscribe to that topic to receive those messages. This creates a many-to-many relationship where multiple nodes can publish to and/or subscribe from the same topic.

### Services
Services provide a request/reply interaction pattern between nodes. A service client sends a request message and waits for a reply message from a service server. This is a one-to-one interaction where only one service server can process requests from multiple service clients.

### Actions
Actions are similar to services but designed for long-running tasks. They support goals that can be sent, feedback that can be received during execution, and results that can be returned upon completion. Actions also support cancellation of goals.

## Role of ROS 2 in Physical AI

ROS 2 serves as the middleware connecting AI logic to humanoid robot control. It provides:

- **Abstraction Layer**: Hides hardware-specific details, allowing AI algorithms to work with standardized interfaces
- **Communication Framework**: Enables different components to exchange data reliably
- **Tool Ecosystem**: Provides debugging, visualization, and development tools
- **Hardware Integration**: Standardized interfaces for sensors and actuators

## Communication Patterns

### Publisher-Subscriber (Topics)
- Asynchronous communication
- Multiple publishers and subscribers can exist for the same topic
- Decouples timing between publishers and subscribers
- Suitable for streaming data like sensor readings

### Client-Server (Services)
- Synchronous communication
- Request-response pattern
- Suitable for operations with clear start and end
- Used for actions that return a result immediately

### Action Server-Client
- Asynchronous with feedback
- Suitable for long-running operations
- Supports goal preemption and cancellation
- Used for navigation, manipulation tasks

## Architecture

ROS 2 uses a DDS (Data Distribution Service) implementation for communication between nodes. This provides:

- **Decentralized Architecture**: No master node required
- **Real-time Support**: Deterministic communication
- **Security**: Built-in security features
- **Multi-language Support**: Python, C++, and other languages

## Summary

ROS 2 fundamentals form the backbone of robotic applications. Understanding nodes, topics, services, and actions is essential for connecting AI logic to robot control systems. These concepts provide the communication infrastructure needed to build complex robotic behaviors.