# Docker Images

This directory contains CrucibleX execution images and deployment helpers.

## Images

- `Dockerfile.cpu` builds a CPU-only image for local and Ray CPU validation.
- `Dockerfile.gpu` builds a GPU image for Ray-backed GPU validation.
- `Dockerfile.npu` builds an Ascend/NPU image for local NPU validation.

All images install CrucibleX with `uv` and expose the `cx` CLI.

## Build

Build from the repository root:

```bash
docker build -f docker/Dockerfile.cpu -t cruciblex-ray-cpu .
docker build -f docker/Dockerfile.gpu -t cruciblex-ray-gpu .
docker build -f docker/Dockerfile.npu -t cruciblex-npu .
```

Override a base image or package index when required:

```bash
docker build -f docker/Dockerfile.npu \
  --build-arg BASE_IMAGE=<ascend-base-image> \
  --build-arg UV_INDEX_URL=<package-index-url> \
  -t cruciblex-npu .
```

## Deployment Notes

- Use `docker/deploy-ray.sh` only for a network-valid CPU/GPU Ray deployment.
- Keep host-specific validation details, private image digests, and environment credentials in `docker/private-notes.md`, which is ignored by git.
- Do not treat this directory as the place for general project guidance; that belongs in `AGENT.md` and the docs under `docs/`.

## Ray Cluster Use

Use built CrucibleX images, not upstream base images, for a Ray cluster. This is valid only where head and workers meet Ray's full bidirectional networking requirement. External clients can use `ray://<head-ip>:10001`; that mode submits work to the accessible Ray cluster and does not turn disconnected client machines into workers.

## SSH Deployment Script

Use `docker/deploy-ray.sh` only for a network-valid CPU/GPU Ray deployment. The current script does not deploy the single-host NPU mode.
