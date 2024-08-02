FROM python:3.9-slim

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev libgl1-mesa-dev nano \
        python3-dev python3 python3-pip \
        make cmake g++ curl build-essential \
        libx11-dev libxext-dev libxrender-dev libglvnd-dev libglib2.0-0\
        libcurl4-openssl-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change the working directory
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .

# Install any dependencies specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for HTTP and HTTPS
EXPOSE 5000

# Define the health check (Dockerfile syntax)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:5000/health || exit 1

# Set the command to run your application
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known_shface:app"]
# Usage: 
# docker build -t facerec_recognize_gpu_ecs:test .
# docker run --gpus all -p 5000:5000 -p 5080:5080 facerec_recognize_gpu_ecs:test

# Slim:
#    slim build \
#    --include-path /usr/local/lib/python3.10/dist-packages \
#    --include-path /usr/bin/nano \
#    --include-path /usr/lib \
#    --keep-perms \
#    --continue-after probe \
#    --target 4980ea56ee27

# docker run -it --rm facerec_recognize_gpu_ecs.slim:latest /bin/sh