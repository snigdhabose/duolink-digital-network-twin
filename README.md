# DuoLink 🔗: Proactive Network Management with Digital Twin Visualization

DuoLink is a proactive network management solution designed to prevent network downtimes, optimize connectivity, and deliver a seamless user experience. By creating a Digital Twin of network infrastructure, DuoLink combines machine learning and reinforcement learning to predict and mitigate network issues in real-time, ensuring uninterrupted service.


markdown
Copy code
## Motivation

### Key Issues:
- **Unexpected Failures**: Hardware faults, configuration errors, and security breaches.
- **Network Congestion**: Traffic spikes leading to degraded performance.
- **Security Threats**: Increasingly sophisticated attacks requiring swift action.

### Reactive vs Predictive vs Proactive Management
- **Reactive Management**: Addresses issues after they occur, leading to downtime and user dissatisfaction.
- **Predictive Management**: Forecasts issues but may not prevent them proactively.
- **Proactive Management**: Anticipates and acts to prevent issues.
  
![image](https://github.com/user-attachments/assets/97e9b5a1-3b88-4168-ae2f-88e6f520a910)

## Key Features

- **Digital Twin Visualization**: A real-time virtual replica of network infrastructure, visualized through Dash, to monitor network performance, traffic flow, and potential failure points.
- **Predictive Maintenance Alerts**: Proactively detects potential network failures and alerts operators.
- **Dynamic Load Balancing**: Manages network traffic to avoid congestion and ensure smooth connectivity.
- **Bandwidth and Resource Optimization**: Allocates network resources efficiently in response to demand.
- **Interactive Monitoring Dashboard**: Displays real-time network status, failure predictions, and proactive actions to keep the network running smoothly.

## Table of Contents

- [Why DuoLink?](#why-duolink)
- [Our Planned Approach](#our-planned-approach)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Demo](#demo)
- [Contributers](#contributers)

## Why DuoLink?

DuoLink addresses critical challenges in network management, such as unexpected failures, network congestion, and security threats. Traditional reactive approaches often result in downtimes, leading to user dissatisfaction and potential revenue loss. DuoLink offers a proactive approach to network management by predicting issues and taking action before they impact users, ensuring continuous connectivity and minimizing disruptions.

### Target Audience

- **Network Administrators**: Receive real-time alerts and proactive insights to prevent network issues before they affect users.
- **Businesses & Service Providers**: Maintain high service quality, reduce downtimes, and protect against revenue loss through reliable network management.

## Our Planned Approach

### Digital Twin
A real-time virtual network replica using NSL-KDD data to simulate traffic, failure points, and security scenarios.

### XGBoost Prediction Model
A predictive model that leverages historical data to identify potential network component failures.

### Reinforcement Learning Agent
Using a **PPO (Proximal Policy Optimization)** RL Agent that adapts network parameters in real time by learning from both predicted and observed network states, making predictions, adjusting routes, bandwidth, and resource allocation.

## Visualization and Testing

Real-time monitoring and predictive insights in **Dash** display network status, failure predictions, and RL adjustments, evaluated by:
- **Prediction Accuracy**
- **Latency Reduction**
- **Minimized Manual Interventions**
  
## Getting Started

### Prerequisites

- **Python 3.7+**
- **Dash**: For real-time data visualization in the monitoring dashboard.
- **Machine Learning Libraries**: XGBoost, TensorFlow, and PPO (Proximal Policy Optimization) for training and deploying predictive models.

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/duolink-network-digital-twin.git
cd duolink-network-digital-twin
pip install -r requirements.txt
```
Ensure that Dash and the required machine learning libraries are properly installed.

### Setup
1. Download the **NSL-KDD dataset** and save it in the `data` folder for training the prediction models.
2. Train the XGBoost model for failure prediction or use the pre-trained model provided in the `models` folder.
3. Configure paths in `config.py` to specify locations for data and model files.

### Usage
To start the DuoLink monitoring dashboard:

```bash
python main.py
```
This command will launch the dashboard, displaying the real-time network status, failure predictions, and suggested reroutes.

![image](https://github.com/user-attachments/assets/efd8a17f-9e3b-4301-8b60-55087632af08)


## Architecture

DuoLink consists of a multi-layered architecture that enables seamless network monitoring and management:

- **Digital Twin Visualization**: Creates a virtual network environment using Dash to monitor real-time network behavior and visualize potential issues.
- **XGBoost Prediction Model**: Utilizes historical data to predict potential network component failures.
- **PPO RL Agent**: A reinforcement learning agent that optimizes network parameters in real time, adjusting traffic flows, rerouting paths, and allocating bandwidth dynamically.
- **Monitoring Dashboard**: An interactive, web-based interface built with Dash that visualizes network health, failure predictions, reroutes, and adjustments.


## Key Components

- **NSL-KDD Dataset**: Provides historical data for training the XGBoost prediction model.
- **Digital Twin Visualization**: Simulates network behavior in a virtual environment using Dash to display real-time data, traffic, and predicted failure points.
- **XGBoost Prediction Model**: Forecasts risks or potential failures in network nodes, providing data for proactive adjustments.
- **PPO RL Agent**: A reinforcement learning agent that adjusts network configurations based on real-time and simulated data to maintain connectivity.
- **Dash Dashboard**: A visualization tool for monitoring network status, predicted failures, and proactive rerouting decisions.

## Demo

To see DuoLink in action, check out our [Demo Video](https://linktodemo.com) and explore the interactive dashboard, including:

- Real-time network monitoring with Digital Twin visualization.
- Proactive failure predictions and automated reroutes.
- Live packet flow visualization to showcase dynamic load balancing and rerouting.

## Contributers
- Snigdha Bose [snigdhab7@gmail.com](mailto:snigdhab7@gmail.com) 
- Kshitij Kumar Srivastava [kshitijswork@gmail.com](mailto:kshitijswork@gmail.com) 

## Contact

For more information, feel free to reach out!
