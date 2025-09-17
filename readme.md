# RL Pong — Playable Pong & Minimal DQN

Two self-contained Python scripts:

- `pong.py` — a polished **playable Pong** built with Pygame (human vs CPU or 2‑player).
- `rl_pong_dqn.py` — a **minimal Deep Q‑Network (DQN)** training skeleton on a lightweight Pong‑like environment (no Gym/ROMs needed).

> Goal: make it dead‑simple to play Pong, then teach an agent to play a simplified version.

---

## Quickstart

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install pygame torch numpy
```
> Tip: For GPU/CUDA builds of PyTorch, follow the official instructions for your platform and Python version.

### 2) Play Pong
```bash
python pong.py
```
**Controls**
- Left paddle: **W/S**
- Right paddle: **Up/Down** (when CPU is off)
- Toggle CPU on/off for right paddle: **C**
- Pause: **P**
- Quit: **ESC** / window close

### 3) Train the RL agent (DQN)
```bash
# Train for N episodes (default=1000)
python rl_pong_dqn.py --train --episodes 1000

# After training, evaluate the saved policy
python rl_pong_dqn.py --play
```
This saves weights to `dqn_pong.pt` in the working directory.

---

## Repository layout
```
.
├── pong.py              # Playable Pong via Pygame (human vs CPU)
├── rl_pong_dqn.py       # Minimal DQN training on a simple Pong-like env
└── dqn_pong.pt          # (created after training) saved model weights
```

---

## What’s inside

### `pong.py` (playable)
- 60 FPS loop, dashed center line, clean UI.
- CPU paddle predicts ball path with a pinch of noise.
- Realistic rebounds: vertical velocity depends on hit position on the paddle.

### `rl_pong_dqn.py` (training)
- **Environment**: `SimplePong` (no graphics). Observation is a 6‑D vector:
  1. Left paddle center y (normalized)
  2. Right paddle center y (normalized)
  3. Ball x (normalized)
  4. Ball y (normalized)
  5. Ball vx (scaled)
  6. Ball vy (scaled)
- **Actions**: `0 = stay`, `1 = up`, `2 = down` (controls the left paddle).
- **Reward**: `+1` when the opponent misses, `-1` when the agent misses (terminal); time limit to end stalemates.
- **Agent**: 2‑layer MLP (128 units each), ε‑greedy, replay buffer, target network, Huber loss, gradient clipping.

---

## How the DQN update works (brief)
For a batch of transitions `(s, a, r, s', done)`:
- Online network estimates `Q(s, ·)`; we pick the value for the taken action: `Q(s, a)`.
- Target network estimates `max_a' Q_target(s', a')`.
- **Bellman target**: `y = r + γ * (1 - done) * max_a' Q_target(s', a')`.
- Optimize Huber loss between `Q(s, a)` and `y`.
- Periodically copy online → target weights.

---

## Key hyperparameters (edit in `train()`)
- `episodes`: total training episodes (CLI `--episodes`).
- `lr = 1e-3`: Adam learning rate.
- `gamma = 0.99`: discount factor.
- `eps_start=1.0, eps_end=0.05, eps_decay=40000`: ε‑greedy schedule.
- `batch_size = 128`: replay minibatch size.
- `sync_every = 1000`: target network sync period (in steps).
- Warmup: learning starts after the buffer holds ≥ 1000 transitions.

**Common tweaks**
- Increase `episodes` for more stable policies.
- Adjust `eps_decay` to maintain exploration longer or shorter.
- Try `sync_every` ∈ {500, 2000} and `batch_size` ∈ {64, 256}.

---

## Troubleshooting
- **Pygame window doesn’t open**: Ensure you’re not in a headless session. Update your video drivers; try running locally.
- **ImportError: torch not found**: Re‑install PyTorch for your Python version. On M1/M2 Macs, use the recommended wheel for your platform.
- **Training looks unstable**: Increase training episodes; try a lower `lr` (e.g., `5e-4`), or increase replay warmup.

---

## Extending this project
- **Better opponents**: Make the CPU imperfect in different ways (reaction delay, capped speed, random blunders).
- **Algorithm upgrades**: Double DQN, Dueling networks, Prioritized Replay, PPO.
- **From pixels**: Replace the 6‑D state with stacked image frames and a ConvNet.
- **Curriculum**: Start with a slower ball or shorter paddle; ramp difficulty over time.
- **Evaluation**: Log episode returns to a CSV and plot learning curves.

---

## FAQ
**Q: Can I watch training?**  
This minimal script trains without rendering for speed. After training, use `--play` to watch the deterministic greedy policy. You can also add logging/plots to visualize learning curves.

**Q: Can I plug the trained agent into `pong.py`?**  
Yes—wire the policy action to control a paddle inside the Pygame loop. (Left paddle uses actions {stay/up/down}).

---

## License
This code is provided for learning and experimentation. Choose and add a license file that suits your use (e.g., MIT).

