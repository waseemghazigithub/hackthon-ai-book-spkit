# Data Model: AI Book Docusaurus Implementation

## Overview

This document defines the data structure and content organization for the AI/Spec-Driven Technical Book implemented with Docusaurus. The book follows a modular approach with multiple modules, starting with Module 1: The Robotic Nervous System (ROS 2).

## Content Structure

### Directory Structure
```
ai_frontend_book/
├── docs/
│   ├── intro.md
│   ├── module1/
│   │   ├── intro.md
│   │   ├── ros2-fundamentals.md
│   │   ├── python-ai-agents.md
│   │   └── urdf-basics.md
│   └── module2/ (future modules)
│       ├── intro.md
│       └── ...
├── src/
│   ├── components/
│   ├── css/
│   └── pages/
├── static/
│   ├── img/
│   └── ...
├── docusaurus.config.js
├── sidebars.js
└── package.json
```

### Content File Schema

Each Markdown file follows this structure:

```markdown
---
title: [Page Title]
description: [Brief description for SEO]
tags: [list, of, relevant, tags]
sidebar_position: [number for ordering in sidebar]
---

# [Main Heading]

[Content body in Markdown format]

## [Subsection 1]

[Subsection content]

## [Subsection 2]

[Subsection content]

### [Sub-subsection]

[More detailed content]

[Code blocks, diagrams, and other content elements]
```

## Module 1: The Robotic Nervous System (ROS 2) Data Model

### Chapter 1: ROS 2 Fundamentals

**File**: `docs/module1/ros2-fundamentals.md`

**Content Elements**:
- Core concepts: nodes, topics, services, actions
- Architecture diagrams
- Communication patterns
- Role in Physical AI
- Code examples (minimal, conceptual)

**Metadata**:
- Title: "ROS 2 Fundamentals"
- Sidebar position: 2
- Tags: ["ros2", "architecture", "communication", "middleware"]

### Chapter 2: Python AI Agents with ROS 2 (rclpy)

**File**: `docs/module1/python-ai-agents.md`

**Content Elements**:
- Python ROS 2 node creation
- Publishing and subscribing patterns
- Service call implementation
- AI agent to robot controller flow
- Code examples using rclpy

**Metadata**:
- Title: "Python AI Agents with ROS 2"
- Sidebar position: 3
- Tags: ["python", "rclpy", "nodes", "publishing", "subscribing", "services"]

### Chapter 3: Humanoid Robot Structure (URDF)

**File**: `docs/module1/urdf-basics.md`

**Content Elements**:
- URDF basics: links, joints, frames
- Humanoid-focused concepts
- URDF role in ROS 2 and simulators
- Example URDF snippets

**Metadata**:
- Title: "Humanoid Robot Structure (URDF)"
- Sidebar position: 4
- Tags: ["urdf", "robotics", "structure", "simulation"]

## Navigation Model

### Sidebar Configuration

The `sidebars.js` file defines the navigation structure:

```javascript
module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro'],
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
    },
  ],
};
```

### Breadcrumb Navigation

Each page will have breadcrumb navigation showing the path:
- Home → Module 1 → [Current Chapter]

## Content Metadata Schema

### Frontmatter Schema

Each Markdown file includes frontmatter with the following properties:

```yaml
---
title: string                    # Required: Page title
description: string              # Optional: SEO description
tags: [string]                   # Optional: Array of tags for categorization
sidebar_position: number         # Optional: Position in sidebar (if different from order in sidebar file)
authors: [string]                # Optional: Array of author names
image: string                    # Optional: Social card image path
keywords: [string]               # Optional: SEO keywords
---
```

## Configuration Model

### Site Configuration (`docusaurus.config.js`)

```javascript
module.exports = {
  // Basic site information
  title: string,
  tagline: string,
  favicon: string,

  // Deployment configuration
  url: string,                    // Base URL for site
  baseUrl: string,                // Base URL path
  organizationName: string,       // GitHub username/organization
  projectName: string,            // GitHub repository name
  deploymentBranch: string,       // Branch for GitHub Pages deployment

  // Feature configuration
  presets: [array],               // Docusaurus presets
  themeConfig: object,            // Theme-specific configuration
  plugins: [array],               // Additional plugins
};
```

## Content Relationships

### Module Dependencies

The content follows a logical progression:
1. **ROS 2 Fundamentals** → Prerequisite for all other chapters
2. **Python AI Agents** → Depends on ROS 2 fundamentals
3. **URDF Basics** → Can be read independently but connects to ROS 2 concepts

### Cross-Module References

Future modules will reference Module 1 concepts:
- Module 2 (Digital Twin) will reference ROS 2 communication patterns
- Module 3 (AI-Robot Brain) will build on Python agent concepts
- Module 4 (VLA) will use URDF knowledge for humanoid control

## Content Validation Rules

### Technical Accuracy
- All code examples must be valid and tested
- Technical concepts must be explained clearly
- Cross-references between chapters must be accurate

### Format Consistency
- All chapters follow the same structural pattern
- Code examples use consistent styling
- Diagrams and images are properly sized and positioned

### Accessibility
- All images have appropriate alt text
- Headings follow proper hierarchy (h1, h2, h3, etc.)
- Content is readable for technical audience