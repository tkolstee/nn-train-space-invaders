#!/usr/bin/env python

import argparse
import random
import os
import shutil
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import ale_py
import ale_py.roms as ale_roms

from cnn import AtariCNN, ReplayBuffer


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 1e-4

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000

NUM_EPISODES = int(os.getenv("NUM_EPISODES", "500"))
MAX_STEPS_PER_EPISODE = int(os.getenv("MAX_STEPS_PER_EPISODE", "20000"))
GAME_ID = os.getenv("GAME_ID", "space_invaders")
RENDER_EVERY = int(os.getenv("RENDER_EVERY", "0"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on an Atari 2600 ROM using ALE.",
    )
    parser.add_argument("--game-id", default=GAME_ID, help="ROM id, e.g. space_invaders")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE, help="Max steps per episode (0=unlimited)")
    parser.add_argument("--render-every", type=int, default=RENDER_EVERY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--replay-size", type=int, default=REPLAY_MEMORY_SIZE)
    parser.add_argument("--target-update", type=int, default=TARGET_UPDATE_FREQ)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-end", type=float, default=EPSILON_END)
    parser.add_argument("--epsilon-decay-steps", type=int, default=EPSILON_DECAY_STEPS)
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save/load checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N episodes")
    parser.add_argument("--save-milestone-every", type=int, default=0, help="Save milestone checkpoints every N episodes (0=disabled)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--play", action="store_true", help="Play mode: load checkpoint and run episodes without training")
    parser.add_argument("--play-episodes", type=int, default=1, help="Number of episodes to play in play mode")
    parser.add_argument("--play-checkpoint", default="best.pt", help="Checkpoint to load in play mode (default: best.pt)")
    return parser.parse_args()


def prepare_local_rom_path(game_id):
    project_dir = Path(__file__).resolve().parent
    game_id = game_id.lower()
    rom_dir = project_dir / ".roms"
    rom_target = rom_dir / f"{game_id}.bin"

    # Check if ROM already exists in .roms/
    if rom_target.exists():
        return rom_target

    # ROM not in .roms/, search project root
    tokens = game_id.replace("_", " ").split()
    rom_candidates = sorted(
        [
            path
            for path in project_dir.glob("*")
            if path.is_file()
            and path.suffix.lower() in {".a26", ".bin"}
            and all(token in path.stem.lower() for token in tokens)
        ]
    )

    if not rom_candidates:
        raise FileNotFoundError(
            f"No ROM found for '{game_id}'. "
            f"Place a ROM file in project root (e.g., 'Space Invaders.a26') or in .roms/ as '{game_id}.bin'."
        )

    # Copy from root to .roms/
    rom_source = rom_candidates[0]
    rom_dir.mkdir(exist_ok=True)
    shutil.copy2(rom_source, rom_target)
    print(f"Copied ROM: {rom_source.name} → .roms/{rom_target.name}")

    return rom_target


def ensure_game_supported(game_id):
    supported_ids = set(ale_roms.get_all_rom_ids())
    if game_id not in supported_ids:
        sample = sorted([rom_id for rom_id in supported_ids if game_id.split("_")[0] in rom_id])
        raise RuntimeError(
            f"This ale-py build does not support '{game_id}' (not in supported ROM IDs). "
            f"Related ROM IDs: {sample}. "
            "Use a supported game ID or different ALE build."
        )


def save_checkpoint(checkpoint_dir, filename, policy_net, target_net, optimizer, episode, global_step, best_reward):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        "policy_net_state_dict": policy_net.state_dict(),
        "target_net_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_reward": best_reward,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_dir, filename, policy_net, target_net, optimizer=None):
    checkpoint_path = Path(checkpoint_dir) / filename
    if not checkpoint_path.exists():
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    target_net.load_state_dict(checkpoint["target_net_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    return {
        "episode": checkpoint["episode"],
        "global_step": checkpoint["global_step"],
        "best_reward": checkpoint["best_reward"],
    }


class AtariRomALEEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        rom_path,
        screen_size=84,
        frame_skip=4,
        frame_stack=4,
        noop_max=30,
        render=False,
    ):
        super().__init__()
        self.rom_path = str(rom_path)
        self.screen_size = int(screen_size)
        self.frame_skip = int(frame_skip)
        self.frame_stack = int(frame_stack)
        self.noop_max = int(noop_max)

        self.ale = ale_py.ALEInterface()
        self.ale.setInt("frame_skip", 1)
        self.ale.setFloat("repeat_action_probability", 0.0)
        self.ale.setBool("display_screen", bool(render))
        self.ale.setBool("sound", bool(render))
        self.ale.loadROM(self.rom_path)

        self.legal_actions = list(self.ale.getLegalActionSet())
        self.action_space = Discrete(len(self.legal_actions))
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_stack, self.screen_size, self.screen_size),
            dtype=np.uint8,
        )
        self.frames = deque(maxlen=self.frame_stack)

    def _get_processed_frame(self):
        frame = self.ale.getScreenGrayscale()
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(frame_tensor, size=(self.screen_size, self.screen_size), mode="area")
        return resized.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ale.reset_game()

        noop_action = 0
        if noop_action not in self.legal_actions:
            noop_action = self.legal_actions[0]

        noop_count = random.randint(0, self.noop_max)
        for _ in range(noop_count):
            self.ale.act(noop_action)
            if self.ale.game_over():
                self.ale.reset_game()

        first_frame = self._get_processed_frame()
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(first_frame)

        obs = np.stack(self.frames, axis=0)
        return obs, {}

    def step(self, action):
        ale_action = self.legal_actions[int(action)]
        total_reward = 0.0

        terminated = False
        for _ in range(self.frame_skip):
            total_reward += float(self.ale.act(ale_action))
            terminated = bool(self.ale.game_over())
            if terminated:
                break

        frame = self._get_processed_frame()
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=0)
        return obs, total_reward, terminated, False, {}

    def close(self):
        pass

def state_to_numpy(observation):
    return np.array(observation, dtype=np.uint8)


def epsilon_by_step(step):
    if step >= EPSILON_DECAY_STEPS:
        return EPSILON_END
    fraction = step / EPSILON_DECAY_STEPS
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def select_action(current_state, eps, env, policy_net):
    if random.random() < eps:
        return env.action_space.sample()

    state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def optimize_model(memory, policy_net, target_net, criterion, optimizer):
    if len(memory) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * max_next_q_values * (1.0 - dones)

    loss = criterion(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def train(checkpoint_dir="checkpoints", save_every=1, save_milestone_every=0, resume=False):
    ensure_game_supported(GAME_ID)
    rom_path = prepare_local_rom_path(GAME_ID)

    def make_env(render=False):
        return AtariRomALEEnv(
            rom_path=rom_path,
            screen_size=84,
            frame_skip=4,
            frame_stack=4,
            noop_max=30,
            render=render,
        )

    train_env = make_env(render=False)
    render_env = make_env(render=True) if RENDER_EVERY > 0 else None

    num_actions = int(train_env.action_space.n)
    memory = ReplayBuffer(capacity=REPLAY_MEMORY_SIZE)

    policy_net = AtariCNN(num_actions).to(device)
    target_net = AtariCNN(num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    start_episode = 0
    global_step = 0
    best_reward = float("-inf")

    if resume:
        checkpoint_data = load_checkpoint(checkpoint_dir, "latest.pt", policy_net, target_net, optimizer)
        if checkpoint_data:
            start_episode = checkpoint_data["episode"] + 1
            global_step = checkpoint_data["global_step"]
            best_reward = checkpoint_data["best_reward"]
            print(f"Resuming from episode {start_episode}, global step {global_step}, best reward {best_reward:.2f}")
        else:
            print(f"No checkpoint found in {checkpoint_dir}, starting fresh")

    for episode in range(start_episode, start_episode + NUM_EPISODES):
        use_render_env = RENDER_EVERY > 0 and (episode + 1) % RENDER_EVERY == 0
        active_env = render_env if use_render_env and render_env is not None else train_env

        obs, _ = active_env.reset()
        state = state_to_numpy(obs)
        episode_reward = 0.0
        done = False
        episode_steps = 0
        epsilon = epsilon_by_step(global_step)
        max_steps_limit = MAX_STEPS_PER_EPISODE if MAX_STEPS_PER_EPISODE > 0 else float("inf")

        while not done and episode_steps < max_steps_limit:
            epsilon = epsilon_by_step(global_step)
            action = select_action(state, epsilon, active_env, policy_net)

            next_obs, reward, terminated, truncated, _ = active_env.step(action)
            done = terminated or truncated
            next_state = state_to_numpy(next_obs)

            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += float(reward)

            optimize_model(memory, policy_net, target_net, criterion, optimizer)

            global_step += 1
            episode_steps += 1
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()

        print(f"Episode {episode + 1}/{start_episode + NUM_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f} | Best: {best_reward:.2f}")

        # Save latest checkpoint
        if (episode + 1) % save_every == 0:
            save_checkpoint(checkpoint_dir, "latest.pt", policy_net, target_net, optimizer, episode, global_step, best_reward)

        # Save milestone checkpoint
        if save_milestone_every > 0 and (episode + 1) % save_milestone_every == 0:
            milestone_filename = f"milestone_ep{episode + 1:06d}.pt"
            save_checkpoint(checkpoint_dir, milestone_filename, policy_net, target_net, optimizer, episode, global_step, best_reward)

        # Save best checkpoint
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_checkpoint(checkpoint_dir, "best.pt", policy_net, target_net, optimizer, episode, global_step, best_reward)

    train_env.close()
    if render_env is not None:
        render_env.close()


def play(checkpoint_dir="checkpoints", checkpoint_file="best.pt", num_episodes=1):
    """Play mode: load a checkpoint and run episodes with rendering, no training."""
    ensure_game_supported(GAME_ID)
    rom_path = prepare_local_rom_path(GAME_ID)

    env = AtariRomALEEnv(
        rom_path=rom_path,
        screen_size=84,
        frame_skip=4,
        frame_stack=4,
        noop_max=30,
        render=True,
    )

    num_actions = int(env.action_space.n)
    policy_net = AtariCNN(num_actions).to(device)
    target_net = AtariCNN(num_actions).to(device)
    policy_net.eval()
    target_net.eval()

    checkpoint_data = load_checkpoint(checkpoint_dir, checkpoint_file, policy_net, target_net, optimizer=None)
    if not checkpoint_data:
        print(f"Error: No checkpoint found at {checkpoint_dir}/{checkpoint_file}")
        return

    print(f"Playing with checkpoint from episode {checkpoint_data['episode']}, best reward {checkpoint_data['best_reward']:.2f}")
    print(f"Running {num_episodes} episode(s) in play mode (no training, epsilon=0)\n")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = state_to_numpy(obs)
        episode_reward = 0.0
        done = False
        episode_steps = 0

        while not done:
            # Pure exploitation (epsilon=0)
            action = select_action(state, eps=0.0, env=env, policy_net=policy_net)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = state_to_numpy(next_obs)
            episode_reward += float(reward)
            episode_steps += 1

        print(f"Play Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {episode_steps}")

    env.close()
    print("\nPlay mode complete.")


if __name__ == "__main__":
    args = parse_args()

    GAME_ID = args.game_id
    NUM_EPISODES = args.num_episodes
    MAX_STEPS_PER_EPISODE = args.max_steps
    RENDER_EVERY = args.render_every
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    REPLAY_MEMORY_SIZE = args.replay_size
    TARGET_UPDATE_FREQ = args.target_update
    LEARNING_RATE = args.learning_rate
    EPSILON_START = args.epsilon_start
    EPSILON_END = args.epsilon_end
    EPSILON_DECAY_STEPS = args.epsilon_decay_steps

    if args.play:
        play(
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_file=args.play_checkpoint,
            num_episodes=args.play_episodes,
        )
    else:
        train(
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            save_milestone_every=args.save_milestone_every,
            resume=args.resume,
        )
