import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Custom sidebar for the AI-Powered Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/intro',
        'module1/ros2-fundamentals',
        'module1/python-ai-agents',
        'module1/urdf-basics',
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/intro',
        'module2/gazebo-physics',
        'module2/unity-interaction',
        'module2/sensor-simulation',
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module3/intro',
        'module3/isaac-sim',
        'module3/isaac-ros',
        'module3/nav2-planning',
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4/intro',
        'module4/voice-commands',
        'module4/llm-planning',
        'module4/capstone-autonomous',
        'module4/tasks',
      ],
      collapsed: true,
    },
  ],
};

export default sidebars;
