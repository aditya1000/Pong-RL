# dashboard.py
import threading
import time
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt

# -----------------------------
# SimplePong environment (no pygame)
# -----------------------------
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
        self.vx = np.random.choice([-1, 1]) * self.BSPEED
        self.vy = np.random.uniform(-1, 1) * self.BSPEED
        self.t = 0
        return self._state()

    def _state(self):
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
        # Agent on left paddle
        if a == 1: self.lp_y -= self.PS
        elif a == 2: self.lp_y += self.PS
        self.lp_y = max(0, min(self.lp_y, self.H - self.PH))

        # Opponent move (simple tracker with noise)
        if self.rp_y + self.PH/2 < self.ball_y - 4:
            self.rp_y += self.PS * 0.9
        elif self.rp_y + self.PH/2 > self.ball_y + 4:
            self.rp_y -= self.PS * 0.9
        self.rp_y += np.random.uniform(-0.5,0.5)
        self.rp_y = max(0, min(self.rp_y, self.H - self.PH))

        # Move ball
        self.ball_x += self.vx
        self.ball_y += self.vy

        # Walls
        if self.ball_y <= 0 or self.ball_y >= self.H:
            self.vy = -self.vy

        # Left paddle collision
        if 10 <= self.ball_x <= 10 + self.PW and self.lp_y <= self.ball_y <= self.lp_y + self.PH and self.vx < 0:
            self.vx = -self.vx
            rel = (self.ball_y - (self.lp_y + self.PH/2)) / (self.PH/2)
            self.vy += rel * 2.0

        # Right paddle collision
        if (self.W - 10 - self.PW) <= self.ball_x <= (self.W - 10) and self.rp_y <= self.ball_y <= self.rp_y + self.PH and self.vx > 0:
            self.vx = -self.vx
            rel = (self.ball_y - (self.rp_y + self.PH/2)) / (self.PH/2)
            self.vy += rel * 2.0

        reward = 0.0
        done = False
        if self.ball_x < -5:
            reward = -1.0; done = True
        elif self.ball_x > self.W + 5:
            reward = +1.0; done = True

        self.t += 1
        if self.t >= 2000:
            done = True

        return self._state(), reward, done, {}

    def render_rgb(self, scale=2):
        # Return an RGB image of the court
        W, H = self.W, self.H
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (25, 25, 30)  # background
        for y in range(0, H, 24):  # center dashed line
            img[y:y+12, W//2-1:W//2+1] = (70, 70, 80)
        # paddles
        y0 = int(self.lp_y); img[y0:y0+self.PH, 10:10+self.PW] = (240, 240, 240)
        y1 = int(self.rp_y); img[y1:y1+self.PH, W-10-self.PW:W-10] = (240, 240, 240)
        # ball
        bx, by = int(self.ball_x), int(self.ball_y)
        img[max(0,by-3):min(H,by+3), max(0,bx-3):min(W,bx+3)] = (120, 200, 255)
        if scale != 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        return img

# -----------------------------
# Q-Network with activation hooks
# -----------------------------
class QNet(nn.Module):
    def __init__(self, in_dim=6, n_actions=3, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)
        self.activations = {}
        self.fc1.register_forward_hook(self._hook('fc1'))
        self.fc2.register_forward_hook(self._hook('fc2'))
        self.fc3.register_forward_hook(self._hook('fc3'))

    def _hook(self, name):
        def h(module, inp, out):
            with torch.no_grad():
                self.activations[name] = out.detach().cpu().numpy()
        return h

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Replay buffer
# -----------------------------
class Replay:
    def __init__(self, cap=50_000):
        self.s = []
        self.cap = cap
    def push(self, item):
        self.s.append(item)
        if len(self.s) > self.cap:
            self.s.pop(0)
    def sample(self, bs):
        idx = np.random.choice(len(self.s), size=bs, replace=False)
        return [self.s[i] for i in idx]

# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class Config:
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 40000
    batch_size: int = 128
    sync_every: int = 1000
    hidden: int = 128
    buffer_warmup: int = 1000

# -----------------------------
# Trainer (threaded loop)
# -----------------------------
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = SimplePong()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(hidden=cfg.hidden).to(self.device)
        self.tgt = QNet(hidden=cfg.hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = Replay()
        self.step_count = 0
        self.stop_flag = threading.Event()
        self.paused = threading.Event()
        self.paused.clear()
        self.returns: List[float] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.last_weights_snapshot: Dict[str, np.ndarray] = {}

    def update_config(self, cfg: Config):
        # live updates for LR and hidden size
        if cfg.lr != self.cfg.lr:
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr
        if cfg.hidden != self.cfg.hidden:
            # rebuild nets if shape changes
            self.q = QNet(hidden=cfg.hidden).to(self.device)
            self.tgt = QNet(hidden=cfg.hidden).to(self.device)
            self.tgt.load_state_dict(self.q.state_dict())
            self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
            self.buf = Replay()
        self.cfg = cfg

    def snapshot_weights(self):
        snap = {}
        for name, p in self.q.named_parameters():
            snap[name] = p.detach().cpu().numpy().copy()
        self.last_weights_snapshot = snap

    def policy(self, s, eps):
        if np.random.rand() < eps:
            return np.random.randint(0, 3)
        with torch.no_grad():
            t = torch.tensor(s, dtype=torch.float32, device=self.device)
            qvals = self.q(t)
            return int(torch.argmax(qvals).item())

    def run(self):
        gamma = self.cfg.gamma
        bs = self.cfg.batch_size
        sync_every = self.cfg.sync_every
        eps_start, eps_end, eps_decay = self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_decay
        device = self.device

        while not self.stop_flag.is_set():
            s = self.env.reset()
            ep_ret = 0.0
            done = False
            while not done and not self.stop_flag.is_set():
                while self.paused.is_set() and not self.stop_flag.is_set():
                    time.sleep(0.05)
                self.step_count += 1
                eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * self.step_count / eps_decay)
                a = self.policy(s, eps)
                s2, r, done, _ = self.env.step(a)
                self.buf.push((s, a, r, s2, float(done)))
                s = s2
                ep_ret += r
                self.epsilons.append(eps)

                if len(self.buf.s) >= self.cfg.buffer_warmup and len(self.buf.s) >= bs:
                    batch = self.buf.sample(bs)
                    bs_t = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
                    ba_t = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                    br_t = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                    bs2_t = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
                    bd_t = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
                    qvals = self.q(bs_t).gather(1, ba_t.view(-1,1)).squeeze(1)
                    with torch.no_grad():
                        tgt_q = self.tgt(bs2_t).max(1).values
                        y = br_t + gamma * (1.0 - bd_t) * tgt_q
                    loss = F.smooth_l1_loss(qvals, y)
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
                    self.opt.step()
                    self.losses.append(float(loss.item()))
                    if self.step_count % 200 == 0:
                        self.snapshot_weights()

                if self.step_count % sync_every == 0:
                    self.tgt.load_state_dict(self.q.state_dict())
            self.returns.append(ep_ret)

    def stop(self):
        self.stop_flag.set()

    def pause(self, flag: bool):
        if flag: self.paused.set()
        else: self.paused.clear()

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(layout="wide", page_title="RL Pong — Live Dashboard")
st.title("RL Pong — Live Training & Visualization")

# Session boot
if "cfg" not in st.session_state:
    st.session_state.cfg = Config()
if "trainer" not in st.session_state:
    st.session_state.trainer = Trainer(st.session_state.cfg)
if "thread" not in st.session_state:
    st.session_state.thread = None

# Sidebar controls
st.sidebar.header("Controls")
run_col1, run_col2, run_col3 = st.sidebar.columns(3)
start = run_col1.button("▶ Start")
pause = run_col2.button("⏸ Pause/Resume")
reset = run_col3.button("⟲ Reset")

st.sidebar.header("Hyperparameters (live)")
lr = st.sidebar.slider("Learning rate", 1e-5, 5e-3, value=st.session_state.cfg.lr, step=1e-5, format="%.5f")
gamma = st.sidebar.slider("Gamma (discount)", 0.85, 0.999, value=st.session_state.cfg.gamma, step=0.001)
eps_decay = st.sidebar.slider("Epsilon decay (steps)", 5_000, 200_000, value=st.session_state.cfg.eps_decay, step=1000)
batch_size = st.sidebar.select_slider("Batch size", options=[32,64,128,256,512], value=st.session_state.cfg.batch_size)
sync_every = st.sidebar.select_slider("Target sync (steps)", options=[250,500,1000,2000,4000], value=st.session_state.cfg.sync_every)
hidden = st.sidebar.select_slider("Hidden units", options=[64,128,256], value=st.session_state.cfg.hidden)

# Update config
new_cfg = Config(lr=lr, gamma=gamma, eps_decay=eps_decay, batch_size=batch_size, sync_every=sync_every, hidden=hidden,
                 eps_start=st.session_state.cfg.eps_start, eps_end=st.session_state.cfg.eps_end, buffer_warmup=st.session_state.cfg.buffer_warmup)
st.session_state.trainer.update_config(new_cfg)
st.session_state.cfg = new_cfg

# Start / Pause / Reset
if start and (st.session_state.thread is None or not st.session_state.thread.is_alive()):
    st.session_state.trainer.stop_flag.clear()
    st.session_state.trainer.paused.clear()
    st.session_state.thread = threading.Thread(target=st.session_state.trainer.run, daemon=True)
    st.session_state.thread.start()

if pause and st.session_state.thread is not None:
    if st.session_state.trainer.paused.is_set():
        st.session_state.trainer.pause(False)
    else:
        st.session_state.trainer.pause(True)

if reset:
    if st.session_state.thread is not None and st.session_state.thread.is_alive():
        st.session_state.trainer.stop()
        time.sleep(0.2)
    st.session_state.trainer = Trainer(st.session_state.cfg)
    st.session_state.thread = None

# Layout: two columns
left, right = st.columns([1,1])

# LEFT: Game rendering (stepped locally for view)
with left:
    st.subheader("Game View")
    mode = st.radio("Agent control mode", ["Policy (ε-greedy)", "Greedy (argmax)"], horizontal=True)
    steps = st.slider("Game steps per refresh", 1, 20, 5, 1)
    trainer = st.session_state.trainer

    for _ in range(steps):
        s = trainer.env._state()
        eps = trainer.epsilons[-1] if trainer.epsilons else 0.1
        a = trainer.policy(s, eps if mode == "Policy (ε-greedy)" else 0.0)
        trainer.env.step(a)
    img = trainer.env.render_rgb(scale=2)
    st.image(img, channels="RGB", caption="SimplePong (agent = left paddle)")

def demo_episode(trainer, max_steps=3000, fps=30, epsilon=0.0):
    """
    Run a single greedy (epsilon=0) episode using the CURRENT trainer.q network,
    and live-stream frames in the app. Uses a fresh SimplePong so training state
    remains untouched.
    """
    env = SimplePong()
    s = env.reset()
    placeholder = st.empty()
    total_reward = 0.0
    for step in range(max_steps):
        # Greedy action (or epsilon-greedy if you set epsilon>0 for experimentation)
        if np.random.rand() < epsilon:
            a = np.random.randint(0, 3)
        else:
            with torch.no_grad():
                t = torch.tensor(s, dtype=torch.float32, device=trainer.device)
                a = int(torch.argmax(trainer.q(t)).item())

        s, r, done, _ = env.step(a)
        total_reward += r

        # Render & display
        frame = env.render_rgb(scale=2)
        placeholder.image(frame, channels="RGB",
                          caption=f"Demo — step {step} • return {total_reward:.2f}")

        # Simple timing to approximate FPS
        if fps and fps > 0:
            time.sleep(1.0 / fps)

        if done:
            break

    st.success(f"Demo finished: steps={step+1}, return={total_reward:.2f}")

    

# RIGHT: Metrics & introspection
with right:
    st.subheader("Training Metrics & Model Introspection")
    m1, m2, m3 = st.columns(3)
    m1.metric("Steps", f"{trainer.step_count}")
    m2.metric("Episodes", f"{len(trainer.returns)}")
    last_eps = trainer.epsilons[-1] if trainer.epsilons else trainer.cfg.eps_start
    m3.metric("ε (exploration)", f"{last_eps:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Episode returns")
        fig1, ax1 = plt.subplots()
        ax1.plot(trainer.returns)
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Return")
        st.pyplot(fig1, clear_figure=True)
    with c2:
        st.caption("Loss (Huber)")
        fig2, ax2 = plt.subplots()
        ax2.plot(trainer.losses[-1000:])
        ax2.set_xlabel("Update"); ax2.set_ylabel("Loss")
        st.pyplot(fig2, clear_figure=True)

    st.markdown("### Weights")
    if trainer.last_weights_snapshot:
        names = sorted([k for k in trainer.last_weights_snapshot.keys() if "weight" in k])
        sel = st.selectbox("Layer weights", names)
        data = trainer.last_weights_snapshot[sel].flatten()
        figw, axw = plt.subplots()
        axw.hist(data, bins=50)
        axw.set_title(f"Histogram of {sel}")
        st.pyplot(figw, clear_figure=True)
    else:
        st.info("Weights snapshot will appear after some training steps.")

    st.markdown("### Activations")
    s = trainer.env._state()
    with torch.no_grad():
        _ = trainer.q(torch.tensor(s, dtype=torch.float32, device=trainer.device))
    if trainer.q.activations:
        layer = st.selectbox("Layer", list(trainer.q.activations.keys()))
        act = trainer.q.activations[layer].flatten()
        figa, axa = plt.subplots()
        axa.hist(act, bins=50)
        axa.set_title(f"Activation distribution — {layer}")
        st.pyplot(figa, clear_figure=True)
    else:
        st.info("No activations captured yet.")
    # --- Demo: watch the current model play a full episode ---
    st.markdown("### Demo: Watch Model Play")
    demo_fps = st.slider("Demo FPS", 10, 60, 30, 1, key="demo_fps")
    demo_eps = 0.0  # set >0 if you want stochasticity in the demo
    if st.button("▶ Play one episode", key="play_demo"):
        # Pause training during the demo to keep things smooth (optional)
        trainer.pause(True)
        try:
            demo_episode(trainer, fps=demo_fps, epsilon=demo_eps)
        finally:
            # Resume training if a training thread is running and not user-paused
            trainer.pause(False)

st.caption("Press ▶ Start to begin training. Adjust sliders live. Use ⏸ to pause/resume.")

if st.sidebar.button("💾 Save model (dqn_pong.pt)"):
    torch.save(st.session_state.trainer.q.state_dict(), "dqn_pong.pt")
    st.sidebar.success("Saved to dqn_pong.pt")


