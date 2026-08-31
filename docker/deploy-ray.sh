#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {build|push|deploy-head|deploy-cpu|deploy-gpu|deploy-all|status|stop}"
  echo ""
  echo "Required env: HEAD_HOST, BUILD_HOST, REGISTRY or IMAGE_CPU/IMAGE_GPU."
  echo "Optional env: CPU_WORKERS, GPU_WORKERS, SSH_USER, SSH_OPTS, BASE_IMAGE_CPU, BASE_IMAGE_GPU."
}

REGISTRY="${REGISTRY:-registry.example.com/cruciblex}"
IMAGE_CPU="${IMAGE_CPU:-${REGISTRY}/cx-ray:2.58.0-py311}"
IMAGE_GPU="${IMAGE_GPU:-${REGISTRY}/cx-ray:2.58.0-py311-cu126}"
BASE_IMAGE_CPU="${BASE_IMAGE_CPU:-rayproject/ray:2.58.0-py311}"
BASE_IMAGE_GPU="${BASE_IMAGE_GPU:-rayproject/ray:2.58.0-py311-gpu}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
BUILD_HOST="${BUILD_HOST:?set BUILD_HOST}"
BUILD_WORKDIR="${BUILD_WORKDIR:-/data2/qszhao2/sunafar/operator/atk_rdg}"

HEAD_HOST="${HEAD_HOST:?set HEAD_HOST}"
CPU_WORKERS="${CPU_WORKERS:-}"
GPU_WORKERS="${GPU_WORKERS:-}"
SSH_USER="${SSH_USER:-root}"
SSH_OPTS="${SSH_OPTS:-}"

HEAD_CONTAINER="${HEAD_CONTAINER:-cx-ray-head}"
CPU_WORKER_NAME="${CPU_WORKER_NAME:-cx-ray-cpu-1}"
GPU_WORKER_NAME="${GPU_WORKER_NAME:-cx-ray-gpu-1}"

HEAD_NUM_CPUS="${HEAD_NUM_CPUS:-0}"
CPU_NUM_CPUS="${CPU_NUM_CPUS:-32}"
GPU_NUM_CPUS="${GPU_NUM_CPUS:-32}"
GPU_NUM_GPUS="${GPU_NUM_GPUS:-8}"
SHM_SIZE="${SHM_SIZE:-10g}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-10001}"
HEAD_NODE_MANAGER_PORT="${HEAD_NODE_MANAGER_PORT:-6380}"
HEAD_OBJECT_MANAGER_PORT="${HEAD_OBJECT_MANAGER_PORT:-6381}"
HEAD_MIN_WORKER_PORT="${HEAD_MIN_WORKER_PORT:-20000}"
HEAD_MAX_WORKER_PORT="${HEAD_MAX_WORKER_PORT:-20099}"
CPU_NODE_MANAGER_PORT="${CPU_NODE_MANAGER_PORT:-6380}"
CPU_OBJECT_MANAGER_PORT="${CPU_OBJECT_MANAGER_PORT:-6381}"
CPU_MIN_WORKER_PORT="${CPU_MIN_WORKER_PORT:-20000}"
CPU_MAX_WORKER_PORT="${CPU_MAX_WORKER_PORT:-29999}"
GPU_NODE_MANAGER_PORT="${GPU_NODE_MANAGER_PORT:-6480}"
GPU_OBJECT_MANAGER_PORT="${GPU_OBJECT_MANAGER_PORT:-6481}"
GPU_MIN_WORKER_PORT="${GPU_MIN_WORKER_PORT:-30000}"
GPU_MAX_WORKER_PORT="${GPU_MAX_WORKER_PORT:-39999}"

ssh_cmd() {
  local host="$1"
  shift
  ssh ${SSH_OPTS} "${SSH_USER}@${host}" "$*"
}

build_images() {
  ssh_cmd "${BUILD_HOST}" "cd ${BUILD_WORKDIR} && docker build -f docker/Dockerfile.cpu --build-arg BASE_IMAGE=${BASE_IMAGE_CPU} --build-arg UV_INDEX_URL=${UV_INDEX_URL} -t ${IMAGE_CPU} ."
  ssh_cmd "${BUILD_HOST}" "cd ${BUILD_WORKDIR} && docker build -f docker/Dockerfile.gpu --build-arg BASE_IMAGE=${BASE_IMAGE_GPU} --build-arg UV_INDEX_URL=${UV_INDEX_URL} -t ${IMAGE_GPU} ."
}

push_images() {
  ssh_cmd "${BUILD_HOST}" "docker push ${IMAGE_CPU} && docker push ${IMAGE_GPU}"
}

deploy_head() {
  ssh_cmd "${HEAD_HOST}" "docker pull ${IMAGE_CPU}; docker rm -f ${HEAD_CONTAINER} >/dev/null 2>&1 || true; docker run -d --name ${HEAD_CONTAINER} --ipc host --net host ${IMAGE_CPU} bash -lc 'ray start --head --node-ip-address=${HEAD_HOST} --port=${RAY_PORT} --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT} --node-manager-port=${HEAD_NODE_MANAGER_PORT} --object-manager-port=${HEAD_OBJECT_MANAGER_PORT} --ray-client-server-port=${RAY_CLIENT_PORT} --min-worker-port=${HEAD_MIN_WORKER_PORT} --max-worker-port=${HEAD_MAX_WORKER_PORT} --num-cpus=${HEAD_NUM_CPUS} --block'"
}

deploy_cpu_worker() {
  local host="$1"
  ssh_cmd "$host" "docker pull ${IMAGE_CPU}; docker rm -f ${CPU_WORKER_NAME} >/dev/null 2>&1 || true; docker run -d --name ${CPU_WORKER_NAME} --shm-size=${SHM_SIZE} --net host ${IMAGE_CPU} bash -lc 'ray start --address=${HEAD_HOST}:${RAY_PORT} --num-cpus=${CPU_NUM_CPUS} --block'"
}

deploy_gpu_worker() {
  local host="$1"
  ssh_cmd "$host" "docker pull ${IMAGE_GPU}; docker rm -f ${GPU_WORKER_NAME} >/dev/null 2>&1 || true; docker run -d --name ${GPU_WORKER_NAME} --shm-size=${SHM_SIZE} --net host --gpus all ${IMAGE_GPU} bash -lc 'ray start --address=${HEAD_HOST}:${RAY_PORT} --num-cpus=${GPU_NUM_CPUS} --block'"
}

for_each_csv_host() {
  local csv="$1"
  local fn="$2"
  IFS="," read -r -a hosts <<< "$csv"
  for host in "${hosts[@]}"; do
    [[ -z "$host" ]] && continue
    "$fn" "$host"
  done
}

status_all() {
  ssh_cmd "${HEAD_HOST}" "docker ps --filter name=${HEAD_CONTAINER} --format '{{.Names}} {{.Status}}'; docker exec ${HEAD_CONTAINER} ray status || true"
  for_each_csv_host "${CPU_WORKERS}" status_cpu
  for_each_csv_host "${GPU_WORKERS}" status_gpu
}

status_cpu() { ssh_cmd "$1" "docker ps --filter name=${CPU_WORKER_NAME} --format '{{.Names}} {{.Status}}'"; }
status_gpu() { ssh_cmd "$1" "docker ps --filter name=${GPU_WORKER_NAME} --format '{{.Names}} {{.Status}}'"; }
stop_cpu() { ssh_cmd "$1" "docker rm -f ${CPU_WORKER_NAME} >/dev/null 2>&1 || true"; }
stop_gpu() { ssh_cmd "$1" "docker rm -f ${GPU_WORKER_NAME} >/dev/null 2>&1 || true"; }

stop_all() {
  for_each_csv_host "${CPU_WORKERS}" stop_cpu
  for_each_csv_host "${GPU_WORKERS}" stop_gpu
  ssh_cmd "${HEAD_HOST}" "docker rm -f ${HEAD_CONTAINER} >/dev/null 2>&1 || true"
}

case "${1:-}" in
  build) build_images ;;
  push) push_images ;;
  deploy-head) deploy_head ;;
  deploy-cpu) for_each_csv_host "${CPU_WORKERS}" deploy_cpu_worker ;;
  deploy-gpu) for_each_csv_host "${GPU_WORKERS}" deploy_gpu_worker ;;
  deploy-all) deploy_head; for_each_csv_host "${CPU_WORKERS}" deploy_cpu_worker; for_each_csv_host "${GPU_WORKERS}" deploy_gpu_worker ;;
  status) status_all ;;
  stop) stop_all ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
