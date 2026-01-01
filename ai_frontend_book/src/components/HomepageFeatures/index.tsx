import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: The Robotic Nervous System (ROS 2)',
    Svg: require('@site/static/img/undraw_ros2_robot.svg').default,
    description: (
      <>
        <p>ROS 2 nodes, topics, services, and actions</p>
        <p>Python AI agents with rclpy</p>
        <p>Humanoid URDF basics</p>
      </>
    ),
  },
  {
    title: 'Module 2: The Digital Twin (Gazebo & Unity)',
    Svg: require('@site/static/img/undraw_gazebo_unity.svg').default,
    description: (
      <>
        <p>Physics and collisions in Gazebo</p>
        <p>High-fidelity interaction in Unity</p>
        <p>Sensor simulation (LiDAR, depth, IMU)</p>
      </>
    ),
  },
  {
    title: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
    Svg: require('@site/static/img/undraw_isaac_ai.svg').default,
    description: (
      <>
        <p>Isaac Sim and synthetic data</p>
        <p>Isaac ROS (VSLAM, navigation)</p>
        <p>Nav2 path planning</p>
      </>
    ),
  },
  {
    title: 'Module 4: Vision-Language-Action (VLA)',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        <p>Voice-to-Action: OpenAI Whisper for voice commands</p>
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className={styles.cardWrapper}>
        <div className={styles.card}>
          <div className="text--center padding-horiz--md">
            <div className={styles.cardIcon}>
              <Svg className={styles.featureSvg} role="img" />
            </div>
            <Heading as="h3" className={styles.cardTitle}>{title}</Heading>
            <p className={styles.cardDescription}>{description}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
