# Ray Image Reference Template

This template shows how to wire the Ray head and worker roles to the CPU and GPU base images.

## Image Map

- Head: `cruciblex-ray-cpu`
- CPU worker: `cruciblex-ray-cpu`
- GPU worker: `cruciblex-ray-gpu`

Use the CPU image for the head node unless you have a separate control-plane image policy.

## Runtime Assumptions

- The image already contains `ray`, `uv`, and the `cx` entrypoint
- The project is installed into the image with `uv pip install --system`
- Worker code and plugins are visible inside the container
- GPU workers have access to the host CUDA runtime and device plugin

## Common Environment

```bash
export CX_BACKEND=cpu
export CX_DEVICE_ID=0
```

Backend-specific workers should override `CX_BACKEND` and `CX_DEVICE_ID` through the runtime layer, not by hand in the application code.

## Head Node

```bash
docker run -d --name cx-ray-head --ipc host --net host cx-ray-cpu bash -lc "ray start --head --node-ip-address=<head-ip> --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --node-manager-port=6380 --object-manager-port=6381 --ray-client-server-port=10001 --min-worker-port=20000 --max-worker-port=20099 --num-cpus=16 --num-gpus=8 --block"
```

## GPU Worker

```bash
docker run -d --name cx-ray-gpu-1 --shm-size=10g --net host --gpus all cx-ray-gpu-cu126 bash -lc "ray start --address=<head-ip>:6379 --node-ip-address=<head-ip> --node-manager-port=6380 --object-manager-port=6381 --min-worker-port=20000 --max-worker-port=29999 --num-cpus=32 --num-gpus=8 --block"

docker run -d --name cx-ray-gpu-2 --ipc host --net host --gpus all cx-ray-gpu-cu126 bash -lc "ray start --address=<head-ip>:6379 --num-cpus=32 --num-gpus=8 --block"
```

## CPU Worker

```bash
docker run -d --name cx-ray-cpu-1 --shm-size=10g --net host cx-ray-cpu bash -lc "ray start --address=<head-ip>:6379 --node-ip-address=<cpu-worker-ip> --node-manager-port=6380 --object-manager-port=6381 --min-worker-port=20000 --max-worker-port=29999 --num-cpus=32 --block"
```



## KubeRay Notes

Use `docs/kuberay-raycluster-template.yaml` as the starting RayCluster manifest.

The template maps:

- head group to the CPU image
- CPU worker group to the CPU image
- GPU worker group to the GPU image with `nvidia.com/gpu: 1`

Keep the project path and plugin path layout stable across all pods. If you build derived CrucibleX images, replace the image fields in the template with those final image tags.

For dashboard access, expose the head service port `8265` through your cluster ingress, NodePort, or port-forward policy.

## Validation Order

1. Start the head node and confirm `cx doctor` works inside the container.
2. Join one CPU worker and confirm Ray sees it.
3. Join one GPU worker and confirm Ray reports the GPU resource.
4. Run `cx run --scheduler ray` against the real cluster.
