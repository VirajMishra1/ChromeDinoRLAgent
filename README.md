# 🦖 Chrome Dino RL Agent: PyTorch

**DQN Chrome Dino Agent: PyTorch**

This project implements a Deep Q-Network (DQN) reinforcement learning agent trained to master a custom PyGame-based Chrome Dino environment. By processing 3D state vectors—dino position, obstacle distance, and obstacle height—the agent optimizes its survival through experience replay and target network synchronization.

---

## 🛠️ Technical Implementation

### 🧠 The Brain (DQN)
The agent utilizes a **Deep Q-Network** architecture built with **PyTorch**:
- **Experience Replay:** Uses a `deque` memory buffer of 5,000 transitions to break temporal correlation and stabilize training.
- **Target Network:** Implements a secondary target model updated every 10 episodes to provide stable Q-value targets.
- **Epsilon-Greedy Strategy:** Features an exploration-exploitation trade-off starting at 1.0 and decaying by 0.99 per step.

### 🎮 The Environment
- **Custom PyGame Engine:** A reconstruction of the classic game optimized for 30 FPS.
- **3D State Vector:** The agent perceives the environment through three key variables:
  1. **Dino Y-Position:** Current vertical coordinate.
  2. **Obstacle Distance:** Horizontal distance between the dino and the nearest obstacle.
  3. **Obstacle Height:** The vertical scale of the approaching threat.

---

## 🏗️ Project Structure
- `dino_ai.py`: PyTorch DQN model and agent logic.
- `dino_game.py`: Custom PyGame environment and physics.
- `train.py`: Training loop and model persistence.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install torch pygame numpy
2. Training
Bash

python train.py
The model will automatically save to dino_ai.pth upon completion of 500 episodes.

Developed by Viraj Mishra
