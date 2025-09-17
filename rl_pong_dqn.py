
"""
Minimal DQN on a lightweight Pong-like environment.

Requirements:
- pip install pygame torch numpy

Usage:
- python rl_pong_dqn.py --train   # trains an agent and saves weights to dqn_pong.pt
- python rl_pong_dqn.py --play    # plays with the trained agent

This is intentionally compact and educational (not SOTA). It uses a low-dimensional
state (ball + paddles). Rendering is optional during training.
"""
import os, math, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Simple Pong-like env (no pygame dependency during training) ---
class SimplePong:
    def __init__(self, width=400, height=300, paddle_h=60, paddle_w=8, ball_size=6, paddle_speed=6, ball_speed=5):
        self.W, self.H = width, height
        self.PH, self.PW = paddle_h, paddle_w
        self.BS = ball_size
        self.PS = paddle_speed
        self.BSPEED = ball_speed
        self.reset()

    def reset(self):
        self.lp_y = self.H/2 - self.PH/2
        self.rp_y = self.H/2 - self.PH/2
        self.ball_x = self.W/2
        self.ball_y = self.H/2
        self.vx = random.choice([-1, 1]) * self.BSPEED
        self.vy = random.uniform(-1, 1) * self.BSPEED
        self.t = 0
        return self._state()

    def _state(self):
        # Normalize positions/velocities to [-1,1]
        s = np.array([
            (self.lp_y + self.PH/2 - self.H/2)/(self.H/2),
            (self.rp_y + self.PH/2 - self.H/2)/(self.H/2),
            (self.ball_x - self.W/2)/(self.W/2),
            (self.ball_y - self.H/2)/(self.H/2),
            self.vx/8.0,
            self.vy/8.0
        ], dtype=np.float32)
        return s

    def step(self, a):
        # a in {0: stay, 1: up, 2: down}
        if a == 1: self.lp_y -= self.PS
        elif a == 2: self.lp_y += self.PS
        self.lp_y = max(0, min(self.lp_y, self.H - self.PH))

        # simple opponent: track ball with noise
        target = self.ball_y - self.PH/2
        if self.rp_y + self.PH/2 < self.ball_y - 4:
            self.rp_y += self.PS * 0.9
        elif self.rp_y + self.PH/2 > self.ball_y + 4:
            self.rp_y -= self.PS * 0.9
        self.rp_y += random.uniform(-0.5,0.5)
        self.rp_y = max(0, min(self.rp_y, self.H - self.PH))

        # move ball
        self.ball_x += self.vx
        self.ball_y += self.vy

        # collide with top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.H:
            self.vy = -self.vy

        # collide with left paddle
        if 10 <= self.ball_x <= 10 + self.PW and self.lp_y <= self.ball_y <= self.lp_y + self.PH and self.vx < 0:
            self.vx = -self.vx
            rel = (self.ball_y - (self.lp_y + self.PH/2)) / (self.PH/2)
            self.vy += rel * 2.0

        # collide with right paddle
        if (self.W - 10 - self.PW) <= self.ball_x <= (self.W - 10) and self.rp_y <= self.ball_y <= self.rp_y + self.PH and self.vx > 0:
            self.vx = -self.vx
            rel = (self.ball_y - (self.rp_y + self.PH/2)) / (self.PH/2)
            self.vy += rel * 2.0

        reward = 0.0
        done = False
        # score conditions
        if self.ball_x < -5:
            reward = -1.0; done = True
        elif self.ball_x > self.W + 5:
            reward = +1.0; done = True

        self.t += 1
        if self.t >= 2000:
            done = True

        return self._state(), reward, done, {}

# --- DQN ---
class QNet(nn.Module):
    def __init__(self, in_dim=6, n_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Replay:
    def __init__(self, cap=50_000):
        self.s = []
        self.cap = cap

    def push(self, item):
        self.s.append(item)
        if len(self.s) > self.cap:
            self.s.pop(0)

    def sample(self, bs):
        return random.sample(self.s, bs)

def train(episodes=1000, render_every=0, save_path="dqn_pong.pt"):
    env = SimplePong()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet().to(device)
    tgt = QNet().to(device)
    tgt.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)
    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 40000
    step_count = 0
    buf = Replay()
    batch_size = 128
    sync_every = 1000

    returns = []
    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            step_count += 1
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * step_count / eps_decay)
            if random.random() < eps:
                a = random.randint(0, 2)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.tensor(s).to(device)).cpu()).item())

            s2, r, done, _ = env.step(a)
            ep_ret += r
            buf.push((s, a, r, s2, done))
            s = s2

            if len(buf.s) >= 1000:
                batch = buf.sample(batch_size)
                bs = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(device)
                ba = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
                br = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
                bs2 = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
                bd = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

                qvals = q(bs).gather(1, ba.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    tgt_q = tgt(bs2).max(1).values
                    y = br + gamma * (1.0 - bd) * tgt_q
                loss = F.smooth_l1_loss(qvals, y)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if step_count % sync_every == 0:
                tgt.load_state_dict(q.state_dict())

        returns.append(ep_ret)
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}: return {np.mean(returns[-50:]):.3f}")

    torch.save(q.state_dict(), save_path)
    print("Saved model to", save_path)

def play(policy_path="dqn_pong.pt", episodes=10, sleep=0.0):
    env = SimplePong()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet().to(device)
    q.load_state_dict(torch.load(policy_path, map_location=device))
    q.eval()

    for ep in range(episodes):
        s = env.reset()
        ret = 0.0
        done = False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(q(torch.tensor(s).to(device))).item())
            s, r, done, _ = env.step(a)
            ret += r
            if sleep > 0:
                time.sleep(sleep)
        print(f"Episode {ep+1} return: {ret:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    if args.train:
        train(episodes=args.episodes)
    elif args.play:
        play()
    else:
        print("Use --train or --play")
