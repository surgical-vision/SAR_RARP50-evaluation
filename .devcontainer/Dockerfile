FROM python:3.10.5-bullseye

COPY requirements.txt /tmp/pip-tmp/
RUN pip3  --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

RUN useradd -ms /bin/bash sar-rarp50
USER sar-rarp50