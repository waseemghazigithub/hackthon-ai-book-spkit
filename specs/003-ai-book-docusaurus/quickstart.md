# Quickstart Guide: AI Book Docusaurus Implementation

## Prerequisites

- Node.js 18.0 or higher
- npm or yarn package manager
- Git for version control

## Setup Instructions

### 1. Clone or navigate to project directory
```bash
cd /path/to/your/project
```

### 2. Create the Docusaurus book directory
```bash
mkdir ai_frontend_book
cd ai_frontend_book
```

### 3. Initialize Docusaurus with classic template
```bash
npx create-docusaurus@latest . classic
```

### 4. Install additional dependencies (if needed)
```bash
npm install @docusaurus/module-type-aliases @docusaurus/types
```

## Development Server

To start the development server:
```bash
cd ai_frontend_book
npm start
```

This will start a local development server at http://localhost:3000

## Adding Module 1 Content

### 1. Create Module 1 directory
```bash
mkdir -p docs/module1
```

### 2. Create Module 1 chapter files
```bash
touch docs/module1/intro.md
touch docs/module1/ros2-fundamentals.md
touch docs/module1/python-ai-agents.md
touch docs/module1/urdf-basics.md
```

### 3. Add content to each chapter file based on the specification

### 4. Update sidebar configuration in `sidebars.js`:
```javascript
module.exports = {
  docs: [
    'intro',
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

## Configuration Changes

### Update `docusaurus.config.js`:
```javascript
module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Connecting AI Systems to Humanoid Robots',
  favicon: 'img/favicon.ico',

  url: 'https://your-username.github.io', // Replace with your GitHub username
  baseUrl: '/ai-book/', // Replace with your repository name
  organizationName: 'your-username', // Replace with your GitHub username
  projectName: 'ai-book', // Replace with your repository name
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-username/your-repo/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
```

## Building for Production

To build the site for production:
```bash
cd ai_frontend_book
npm run build
```

The built site will be in the `build/` directory.

## Deploying to GitHub Pages

### 1. Configure deployment in `docusaurus.config.js` (as shown above)

### 2. Create GitHub Actions workflow at `.github/workflows/deploy.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  deploy:
    name: Deploy Docusaurus
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
          cache: npm
          cache-dependency-path: ai_frontend_book/package-lock.json

      - name: Install dependencies
        run: npm ci
        working-directory: ai_frontend_book

      - name: Build website
        run: npm run build
        working-directory: ai_frontend_book

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./ai_frontend_book/build
          publish_branch: gh-pages
```

## Running Tests

Docusaurus includes built-in test capabilities:
```bash
cd ai_frontend_book
npm test
```

## Customization Options

### Adding Custom Components
Create components in `src/components/` and import them in your Markdown files:

```md
import ComponentName from '@site/src/components/ComponentName';
```

### Adding Static Assets
Place static assets in the `static/` directory. They will be served at the root path:
- `static/img/logo.svg` becomes available at `/img/logo.svg`

### Custom CSS
Add custom styles in `src/css/custom.css` which is loaded by default.