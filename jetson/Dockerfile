# JetPack 6.2 base (requires: docker login nvcr.io)
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# --- base OS + tools ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales tzdata curl wget ca-certificates gnupg2 lsb-release \
    python3-pip python3-opencv git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
 
# Make sure bash is used so we can "source" things in RUN steps if needed
SHELL ["/bin/bash", "-lc"]

# After apt installs:
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir "packaging>=24.1" "setuptools<80" wheel --break-system-packages
 

#    (Fixes libopenblas and OpenCV)
# 2. Install System Dependencies (Fixes libopenblas and OpenCV)
#    - libopenblas-dev: Fixes the PyTorch crash
#    - python3-opencv: Installs the CUDA-accelerated OpenCV pre-built for Jetson
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libopenmpi-dev \
    python3-opencv \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade Pip (Important for aarch64 wheels)
RUN pip3 install --upgrade pip

# 4. Install Python Packages
#    - transformers: For your ViT model
#    - numpy<2.0: Fixes the "NumPy 2.0" conflict 
RUN pip3 install \
    transformers \
    "numpy<2.0"

# 5. (Optional) Set working directory
WORKDIR /workspace/ros2_ws
# Add ROS 2 apt repo & key
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && add-apt-repository universe && apt-get update && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    . /etc/os-release && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      python3-colcon-common-extensions \
      python3-rosdep python3-argcomplete \
      ros-humble-rclpy ros-humble-std-msgs ros-humble-geometry-msgs ros-humble-sensor-msgs \
      ros-humble-v4l2-camera ros-humble-cv-bridge ros-humble-image-transport ros-humble-vision-msgs \
      libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
RUN rosdep init || true && rosdep update || true

# Initialize rosdep (inside container)
RUN rosdep init || true && rosdep update

# --- PyTorch (CUDA 12.6 for JP6) ---
# Use the JP6/cu126 wheels index for GPU-enabled torch
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

#  extras
# RUN pip3 install transformers==4.44.2 timm==1.0.9 numpy==1.26.4
# ROS packages used by face_id_ros camera pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libopencv-dev \
    ros-humble-v4l2-camera \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

 

# (optional) helpful tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano htop iputils-ping \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg alsa-utils portaudio19-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install python-dotenv google-generativeai textblob elevenlabs SpeechRecognition pyaudio

# Workspace layout
WORKDIR /workspace



# Keep an entrypoint handy ( mount code at runtime)
COPY jetson/entrypoint.sh /opt/cara/entrypoint.sh
RUN chmod +x /opt/cara/entrypoint.sh


# Audio & system runtime deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    flac \
    mpg123 \
    portaudio19-dev \
    pulseaudio \
    libportaudio2 \
    alsa-utils \
    ffmpeg \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Python deps for voice / TTS / STT (playsound is a Python package) ----
# Prefer a requirements.txt if you have one; otherwise:
RUN python3 -m pip install --no-cache-dir \
    python-dotenv \
    SpeechRecognition \
    pyaudio \
    pydub \
    playsound==1.2.2 \
    google-generativeai \
    elevenlabs \
    textblob
RUN python3 -m pip install --no-cache-dir \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor


# Auto-source ROS and workspace for interactive shells
RUN echo 'if [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; fi' >> /root/.bashrc && \
    echo 'if [ -f /workspace/ros2_ws/install/setup.bash ]; then source /workspace/ros2_ws/install/setup.bash; fi' >> /root/.bashrc

RUN echo 'export COLCON_PYTHON_SETUP_PY_STRATEGY=install' >> /root/.bashrc

# Default: check CUDA & drop to a shell if no launch args
ENTRYPOINT ["/opt/cara/entrypoint.sh"]
#  
# CMD ["/opt/cara/entrypoint.sh"]
