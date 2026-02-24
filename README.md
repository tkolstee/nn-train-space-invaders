# DQN Space Invaders Training

Deep Q-Network (DQN) agent for Atari 2600 Space Invaders using PyTorch and ALE.

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Place your ROM in project root (e.g., "Space Invaders.a26")
# First run will copy it to .roms/space_invaders.bin

# Train for 500 episodes
python main.py --num-episodes 500

# Resume training
python main.py --resume --num-episodes 500

# Watch the best checkpoint play
python main.py --play --play-checkpoint best.pt
```

## Docker Usage

### Build the Image

```bash
docker build -t dqn-space-invaders .
```

### Prepare ROMs

Before running, ensure your ROM is available:

```bash
# Option 1: Copy ROM to .roms/ directory with correct name
mkdir -p .roms
cp "Space Invaders.a26" .roms/space_invaders.bin

# Option 2: Build with ROM in context and let the script copy it
# (Place ROM file in project root before building)
```

### Run Training

```bash
# Create host directory for checkpoints
mkdir -p ./checkpoints

# Run training with checkpoints saved to host
docker run --rm \
  -v "$(pwd)/.roms:/app/.roms:ro" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  dqn-space-invaders \
  --num-episodes 1000 \
  --save-every 10 \
  --save-milestone-every 100

# Resume training from checkpoint
docker run --rm \
  -v "$(pwd)/.roms:/app/.roms:ro" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  dqn-space-invaders \
  --resume --num-episodes 1000
```

### View Progress

While training runs on a server, periodically copy `checkpoints/` to your laptop:

```bash
# On server: checkpoint files are in ./checkpoints/
# - best.pt (best performing model)
# - latest.pt (most recent, use for --resume)
# - milestone_ep000100.pt, milestone_ep000200.pt, etc. (intermediate progress)

# On laptop: copy checkpoint + ROM, then play
python main.py --play --play-checkpoint best.pt
python main.py --play --play-checkpoint milestone_ep000500.pt
```

## CLI Options

### Training Options
- `--game-id` - ROM id (default: space_invaders)
- `--num-episodes` - Number of episodes to train (default: 500)
- `--max-steps` - Max steps per episode (default: 20000, `0` = unlimited/run until game over)
- `--resume` - Resume from latest checkpoint

### Checkpoint Options
- `--checkpoint-dir` - Directory for checkpoints (default: checkpoints)
- `--save-every` - Save latest.pt every N episodes (default: 1)
- `--save-milestone-every` - Save milestone_epXXXXXX.pt every N episodes (default: 0=disabled)

### Play/Evaluation Options
- `--play` - Play mode (no training, just run episodes)
- `--play-episodes` - Number of episodes in play mode (default: 1)
- `--play-checkpoint` - Checkpoint file to load (default: best.pt)

### Hyperparameters
- `--batch-size` - Batch size for training (default: 32)
- `--gamma` - Discount factor (default: 0.99)
- `--learning-rate` - Learning rate (default: 1e-4)
- `--epsilon-start` - Starting exploration rate (default: 1.0)
- `--epsilon-end` - Final exploration rate (default: 0.05)
- `--epsilon-decay-steps` - Steps to decay epsilon (default: 100000)

### Rendering
- `--render-every` - Render every Nth episode during training (default: 0=disabled)

## Checkpoint Files

- **latest.pt** - Most recent checkpoint, updated every `--save-every` episodes. Use this with `--resume`.
- **best.pt** - Checkpoint with highest episode reward so far. Use this for evaluation/play.
- **milestone_epXXXXXX.pt** - Intermediate checkpoints saved every `--save-milestone-every` episodes for viewing training progression.

Checkpoints are portable across CPU architectures (x86 ↔ ARM/M1/M2).

## Example: Long Training Run

```bash
# Server: Train for 10,000 episodes, save milestones every 500 episodes
docker run -d --name dqn-train \
  -v "$(pwd)/.roms:/app/.roms:ro" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  dqn-space-invaders \
  --num-episodes 10000 \
  --save-every 10 \
  --save-milestone-every 500

# Check progress
docker logs dqn-train

# Laptop: Copy checkpoints and view intermediate progress
scp -r server:~/project/checkpoints ./
python main.py --play --play-checkpoint milestone_ep001000.pt
python main.py --play --play-checkpoint milestone_ep002000.pt
python main.py --play --play-checkpoint best.pt
```
