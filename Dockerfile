# Use a more specific Ubuntu version for better reproducibility
FROM ubuntu:22.04

# Set environment variables to improve Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES="-1" \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install Python and pip, and clean up in the same layer to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /opt/venv
USER appuser

# Expose the application port
EXPOSE 5000

# Set the default command to run the application
CMD ["python3", "server.py"]