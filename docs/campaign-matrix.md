# Campaign Matrix 组合策略

CrucibleX 使用 `campaign.matrix` 展开 operator、backend、task、dtype、shape 等运行维度。它替代传统 traversal 命令的静态组合部分，并保留普通 `runs` 与 matrix 扩展结果并存的能力。

## 完整展开

```yaml
runs:
  - name: smoke
    case: examples/cases/torch.relu.tri-accuracy.yaml
    nodes: examples/nodes/local-cpu-gpu.yaml
    task: accuracy
matrix:
  base:
    case: examples/cases/torch.relu.tri-accuracy.yaml
    nodes: examples/nodes/local-cpu-gpu.yaml
    task: accuracy
  dimensions:
    dtype: [fp16, fp32]
    shape: [[1, 32], [4, 64]]
```

维度按键名排序后进行笛卡尔展开，每个组合都会获得稳定的 `matrix_id`。

## 限制规模

大矩阵可以使用 `max_runs` 和 `seed` 进行确定性抽样：

```yaml
matrix:
  base:
    case: examples/cases/torch.relu.tri-accuracy.yaml
    nodes: examples/nodes/local-cpu-gpu.yaml
    task: accuracy
  dimensions:
    dtype: [fp16, fp32, fp64]
    shape: [[1, 32], [4, 64], [8, 128]]
  max_runs: 4
  seed: 11
```

当完整组合数大于 `max_runs` 时，CX 使用 seed 驱动的稳定抽样选择组合，并在每个选择结果中记录：

- `matrix_total`：完整矩阵组合数；
- `matrix_selected`：当前实际选择的组合数；
- `matrix_id`：组合内容的稳定标识。

campaign 的 `--shard-index` / `--shard-count` 在 matrix 抽样之后执行。因此可先限制整体规模，再把同一稳定选择集分发到多个 worker。`max_runs` 必须为正整数；未声明时保持完整笛卡尔展开语义。
