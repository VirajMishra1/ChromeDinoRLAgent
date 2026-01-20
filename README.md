# Dino DQN (Pygame) 🦖🤖

A minimal **Chrome Dino–style runner** built in **Pygame**, trained with a **Deep Q-Network (DQN)** agent in **PyTorch**.

The agent observes a small state vector (dino height + obstacle distance) and learns when to **jump** to avoid obstacles.

---

## What’s inside

- **`dino_game.py`** — Pygame runner environment (dino + obstacles + score)
- **`dino_ai.py`** — DQN model + replay buffer + epsilon-greedy policy + target network
- **`train.py`** — training loop (renders the game while training) and saves weights to `dino_ai.pth`

---

## How the game works

**Game window:** `600 x 200`  
**Dino:** rectangle at `(x=50, y=130)` with size `30 x 60`  
**Obstacles:** red rectangles `20 x 20`, move left by `5 px/frame`

### Actions
- `0` → do nothing  
- `1` → jump (only if currently on the ground)

### State (input to the network)
A 3D vector:

```
[dino_y, obstacle_distance, obstacle_height]
```

- If an obstacle exists: `obstacle_distance = obstacles[0].x - dino.x`, `obstacle_height = 20`
- If none: `[dino_y, 999, 0]`

### Rewards
- `+1` each step you survive
- `-1` on collision (episode ends)

---

## How the AI works (DQN)

**Network:** `3 → 24 → 24 → 2` (ReLU activations)

Key training pieces:
- **Replay buffer:** up to `5000` transitions
- **Epsilon-greedy:** starts at `1.0`, decays by `0.99` each replay step, min `0.05`
- **Discount factor (gamma):** `0.95`
- **Target network:** synced periodically for stability (`UPDATE_TARGET_EVERY = 10` episodes)

Trained weights are saved to:

```
dino_ai.pth
```

---

## Setup

### Requirements
- Python 3.8+ recommended
- `pygame`
- `torch`
- `numpy`

Install dependencies:

```bash
pip install pygame torch numpy
```

---

## Train the agent

Run:

```bash
python train.py
```

Notes:
- Training **opens a Pygame window** and renders gameplay while learning.
- Close the window to stop training early.
- After training finishes, weights are saved as `dino_ai.pth`.

You’ll see logs like:

```
Episode: 12, Score: 4, Epsilon: 0.73
```

---

## Loading a trained model (quick snippet)

There isn’t a dedicated “play with trained model” script in this repo, but you can do it with something like:

```python
import torch
from dino_game import DinoGame
from dino_ai import DinoAI

game = DinoGame()
ai = DinoAI()
ai.model.load_state_dict(torch.load("dino_ai.pth", map_location="cpu"))
ai.epsilon = 0.0  # greedy

state = game.get_state()
while game.running:
    game.render()
    action = ai.act(state)
    reward = game.update(action)
    state = game.get_state()
    if reward == -1:
        break

print("Final score:", game.score)
```

---

## Tweaking / Ideas to improve learning

If you want to push performance further:
- Add more state info (e.g., obstacle width, speed, multiple obstacles)
- Use frame stacking or normalized inputs
- Reward shaping (e.g., score-based reward)
- Increase network size and/or train longer
- Add a “no render” mode for faster training (render only occasionally)

---

## Troubleshooting

- **Window freezes / not responding:** this project handles `pygame.QUIT` events inside `render()`. Keep `render()` inside the loop (as in `train.py`).
- **Slow training:** rendering is the bottleneck. Consider disabling rendering during most episodes.

---

## License

Use freely for learning / demos. Add a license file if you plan to publish publicly.
