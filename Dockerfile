FROM python:3.10.5-bullseye


# this is for opencv, we could do headless but autocomplete does not work
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     && rm -rf /var/lib/apt/lists/*

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp
# RUN pip3  --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp



# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

RUN useradd -ms /bin/bash sar-rarp50
USER sar-rarp50