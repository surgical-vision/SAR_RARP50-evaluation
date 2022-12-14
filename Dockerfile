FROM python:3.10.5-bullseye

COPY requirements.txt /tmp/pip-tmp/
RUN pip3  --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

RUN useradd -ms /bin/bash sar-rarp50
RUN mkdir workspace && chown sar-rarp50 /workspace
USER sar-rarp50
WORKDIR ./workspace


COPY . .


ENTRYPOINT [ "python", "-m", "scripts.sarrarp50"]