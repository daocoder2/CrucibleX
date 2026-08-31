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

## NPU 一次性验证

CANN 8.3 RC1 镜像已包含自身的 CANN runtime。运行时挂载宿主 Ascend driver、暴露必要 NPU 设备节点，并使用 host network 和 IPC。构建完成的 CrucibleX 镜像不应挂载宿主 CANN toolkit 或源码目录。

```bash
docker run --rm --privileged --net=host --ipc=host \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$(pwd)/cx_output:/out" \
  <cruciblex-npu-image> \
  bash -lc 'python -c "import torch; import torch_npu; assert torch.npu.is_available()"; OUTPUT_ROOT=/out bash scripts/npu_aclnn_array_gate.sh'
```

NPU 镜像会打包 `scripts/`，gate 直接在构建镜像内执行。不要把 CANN `aarch64-linux/devlib` 前置到 `LD_LIBRARY_PATH`：它可能覆盖镜像选择的 driver-compatible runtime，使 `torch.npu` 不可用。

## Deployment Notes

- Use `docker/deploy-ray.sh` only for a network-valid CPU/GPU Ray deployment.
- Keep host-specific validation details, private image digests, and environment credentials in `docker/private-notes.md`, which is ignored by git.
- Do not treat this directory as the place for general project guidance; that belongs in `AGENT.md` and the docs under `docs/`.

## Ray Cluster Use

Use built CrucibleX images, not upstream base images, for a Ray cluster. This is valid only where head and workers meet Ray's full bidirectional networking requirement. External clients can use `ray://<head-ip>:10001`; that mode submits work to the accessible Ray cluster and does not turn disconnected client machines into workers.

## SSH Deployment Script

Use `docker/deploy-ray.sh` only for a network-valid CPU/GPU Ray deployment. The current script does not deploy the single-host NPU mode.
