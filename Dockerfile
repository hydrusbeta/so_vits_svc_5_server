# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies.
# Binaries are not available for some python packages, so pip must compile them locally. This is
# why gcc, g++, and python3.8-dev are included in the list below.
# Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute capability
# 3.5 through 9.0
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone.
RUN apt update && apt install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    python3.8-dev \
    python3.8-venv \
    python3.9-venv \
    wget \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg

# Create virtual environments for so-vits-svc 5.0 and Hay Say's so_vits_svc_5_server
RUN python3.8 -m venv ~/hay_say/.venvs/so_vits_svc_5; \
    python3.9 -m venv ~/hay_say/.venvs/so_vits_svc_5_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517
RUN ~/hay_say/.venvs/so_vits_svc_5/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel; \
    ~/hay_say/.venvs/so_vits_svc_5_server/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel

# Install all python dependencies for so-vits-svc 5.0.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# so-vits-svc 5.0 code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build
# time. See https://docs.docker.com/build/cache
RUN ~/hay_say/.venvs/so_vits_svc_5/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    fsspec==2023.5.0 \
    ffmpeg-python==0.2.0 \
    praat-parselmouth==0.4.3 \
    pyworld==0.3.4 \
    matplotlib==3.7.1 \
    soundfile==0.12.1 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    torchcrepe==0.0.19 \
    tensorboard==2.13.0 \
    transformers==4.29.2 \
    tqdm==4.65.0 \
    librosa==0.9.1 \
    omegaconf==2.3.0

# Install the dependencies for the Hay Say interface code
RUN ~/hay_say/.venvs/so_vits_svc_5_server/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    hay_say_common==1.0.1 \
    jsonschema==4.17.3

# Download the Timbre Encoder
RUN mkdir -p ~/hay_say/temp_downloads/hubert/ && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E" -O /root/hay_say/temp_downloads/hubert/best_model.pth.tar && rm -rf /tmp/cookies.txt

# Download the pretrained Whisper model
RUN mkdir -p ~/hay_say/temp_downloads/whisper_pretrain/ && \
    wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt --directory-prefix=/root/hay_say/temp_downloads/whisper_pretrain/

# Expose port 6577, the port that Hay Say uses for so-vits-svc 5.0
EXPOSE 6577

# download so-vits-svc 5.0 and checkout a specific commit that is known to work with this Docker file and with Hay Say
RUN git clone -b bigvgan-mix-v2 --single-branch -q https://github.com/PlayVoice/so-vits-svc-5.0 ~/hay_say/so_vits_svc_5
WORKDIR /root/hay_say/so_vits_svc_5
RUN git reset --hard 8f5b747a48f7a67cc8dcd275fcf5ddfb34d38ff3

# Move the Timbre Encoder and Whisper model to the expected directories:
RUN mv /root/hay_say/temp_downloads/hubert/best_model.pth.tar /root/hay_say/so_vits_svc_5/hubert/ && \
    mv /root/hay_say/temp_downloads/whisper_pretrain/medium.pt /root/hay_say/so_vits_svc_5/whisper_pretrain/

# Create directories that are used by the Hay Say interface code
RUN mkdir -p /root/hay_say/so_vits_svc_5/input; \
    mkdir -p /root/hay_say/so_vits_svc_5/ppg; \
    mkdir -p /root/hay_say/so_vits_svc_5/pit

# Download the Hay Say interface code
RUN git clone https://github.com/hydrusbeta/so_vits_svc_5_server ~/hay_say/so_vits_svc_5_server/

# Run the Hay Say interface on startup
CMD ["/bin/sh", "-c", "/root/hay_say/.venvs/so_vits_svc_5_server/bin/python /root/hay_say/so_vits_svc_5_server/main.py --cache_implementation file"]
