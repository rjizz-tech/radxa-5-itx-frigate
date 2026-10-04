FROM ghcr.io/blakeblackshear/frigate:0.18.0-rk@sha256:88e8d411a06fee0926e4fa1d86a8bd48f14256901672f758c8747131e39e7925
ARG TARGETARCH
RUN test "${TARGETARCH}" = "arm64"
LABEL org.opencontainers.image.title="ROCK 5 ITX Frigate AX-M1" \
      org.opencontainers.image.version="0.18.0-radxa.1" \
      org.opencontainers.image.description="RKMPP video, AXCL object/face inference, RAM segment staging"
COPY docker /opt/radxa/docker
COPY overlay /opt/radxa/overlay
COPY tools /opt/radxa/tools
COPY models/manifest.json /opt/radxa/models-lock.json
RUN python3 /opt/radxa/docker/patch_upstream.py \
    && python3 -m compileall -q /opt/frigate/radxa_frigate /opt/frigate/frigate/detectors/plugins/axengine.py \
    && python3 -c "import importlib.metadata; assert importlib.metadata.version('axengine') == '0.1.3'"
ENV TZ=Australia/Melbourne \
    RADXA_LOCK_DIR=/dev/shm/radxa-axcl \
    RADXA_STATE_DIR=/dev/shm/radxa-status
