# 失败重现产物

`cx repro` 生成的 bundle 按 failure cluster 保存失败证据、固定输入和可重放 Case。

每个 cluster 目录包含：

- `failure.json`：原始失败聚类和来源结果；
- `inputs.json`：代表性失败的输入快照；
- `minimized_case.yaml`：使用 `dump_replay` 固定读取 bundle 内输入的 Case；
- `repro.sh`：执行该独立 Case 的重放脚本；
- `semantic_reduction.yaml` 或 `semantic_reduction_candidate.yaml`：可选语义规约结果。

`minimized_case.yaml` 的生成策略会覆盖原有随机 generator，设置 `generator: dump_replay` 和 `generation.metadata.input_snapshot_path`，因此重放不会重新采样输入。脚本会调用 bundle 内的 Case，并将结果写入当前 cluster 的 `rerun-output` 目录。

语义规约只有在提供 `replay_predicate`（通常来自 `--replay-command`）时才会被验证。没有 predicate 时，CX 只生成候选，不将候选宣称为已复现的最小 Case。

```bash
cx repro --output cx_output --cluster-id cluster-0 --minimize --script
./cx_output/repro/<cluster>/repro.sh
```

重放脚本仍使用原运行的 node 配置、scheduler、Ray 地址和插件路径；这些运行环境不是 bundle 内的输入数据，因此跨机器重放时需要提供等价运行环境。
