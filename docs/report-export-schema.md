# Report Export Schema

CrucibleX 同时保留原始执行结果与版本化报告投影：

- `results.jsonl` / `results.csv`：执行结果原始序列化，保留动态 `metrics`、artifacts 和后端扩展；
- `report.jsonl` / `report.csv`：面向 CI、趋势系统和数据仓库的稳定字段投影。

稳定投影使用 `report_schema_version: 1`。新增字段只会以向后兼容方式添加；字段语义发生不兼容变化时必须提升 schema version。

## 核心字段域

- 身份与来源：`run_id`、`plan_id`、`case_id`、`case_name`、`case_fingerprint`、`matrix_id`、`manifest_lane`、`manifest_lane_kind`、`manifest_case_include`、fuzz generation 信息；
- 执行：`task`、`status`、`node_name`、`backend`、`device_id`、`resolved_device`、`candidate_executor`；
- 硬件证据：`hardware_probe_status`、`hardware_backend`、`hardware_device_id`、`hardware_fingerprint`、`hardware_runtime_json`；
- 精度：`comparison`、最大/平均绝对误差、最大/平均相对误差、`rmse`、`matched_ratio`；
- 性能与内存：`latency_ms`、`throughput_items_per_s`、`memory_peak_bytes`；
- 失败与环境：`failure_kind`、`failure_stage`、`error`、ACLNN capability/status/reason/decisions；
- 可扩展证据：`runtime_policy_json`、`manifest_runtime_json`、`metrics_json`、`evidence_json`。

复杂对象统一以 JSON 字符串放进 `*_json` 字段，因此 CSV 列顺序不受 backend、comparator 或 plugin 的动态 metrics 影响。`aclnn_capability_decisions_json` 保留 spec 级完整 ABI 判定列表，单项 `aclnn_capability/status/reason` 仍指向首个阻塞判定，便于快速筛查与向后兼容读取。`hardware_runtime_json` 保留硬件 probe 原始运行时摘要，`hardware_backend` / `hardware_device_id` / `hardware_fingerprint` 则保留硬件身份投影。JSONL 使用相同字段名和同一版本号。manifest lane 字段来自 case metadata 投影，未通过 manifest 运行时为空字符串。

每次 `cx run` 都会生成：

```text
<output>/results.jsonl     # 原始执行记录
<output>/results.csv       # 原始扁平记录
<output>/report.jsonl      # report_schema_version = 1
<output>/report.csv        # 与 report.jsonl 相同字段顺序
```

执行 `cx report` 后会额外生成 `<output>/report.md`，其中标注稳定导出路径与 schema 版本。
