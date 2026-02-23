FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py cnn.py ./

# Create directories for ROMs and checkpoints
RUN mkdir -p .roms checkpoints

# The .roms directory should contain your ROM file
# The checkpoints directory will be bind-mounted
VOLUME ["/app/checkpoints"]

# Default command: run training with checkpoints saved to bind-mounted volume
CMD ["python", "main.py", "--checkpoint-dir", "/app/checkpoints"]
