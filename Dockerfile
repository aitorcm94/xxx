# Ultra-optimized multi-stage build
FROM python:3.11-slim as base-builder

# Install only absolutely necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgcc-12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

# Install PyTorch CPU-only first (smallest version)
RUN pip install --user --no-cache-dir torch==2.5.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Remove unnecessary files from pip install
RUN find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "*.pyo" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Ultra-minimal runtime stage using distroless-like approach
FROM gcr.io/distroless/python3-debian12:nonroot

# Copy only the Python packages we need
COPY --from=base-builder /root/.local /home/nonroot/.local

# Set environment for distroless
ENV PATH="/home/nonroot/.local/bin:$PATH"
ENV PYTHONPATH="/home/nonroot/.local/lib/python3.11/site-packages"

# Copy application code
COPY --chown=nonroot:nonroot app/ /app/

# Set working directory
WORKDIR /app

# Use distroless nonroot user (already configured)
USER nonroot

ENTRYPOINT ["python", "-m", "app.main"]