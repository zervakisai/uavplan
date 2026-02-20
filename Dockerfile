FROM python:3.12-slim

LABEL maintainer="Konstantinos Zervakis"
LABEL description="UAVBench — Urban UAV Path-Planning Benchmark"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY tests/ tests/
COPY data/ data/
COPY scripts/ scripts/
COPY tools/ tools/
COPY docs/ docs/

# Install package
RUN pip install --no-cache-dir -e ".[dev]"

# Verify installation
RUN python -m pytest tests/ -q --tb=short

# Default: run smoke test
CMD ["python", "-m", "uavbench", \
     "--mode", "nav", \
     "--scenarios", "gov_civil_protection_easy", \
     "--planners", "astar", \
     "--trials", "1"]
