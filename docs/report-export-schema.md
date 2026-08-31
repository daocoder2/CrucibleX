# Report Export Schema

CrucibleX 同时保留原始执行结果与版本化报告投影：

- `results.jsonl` / `results.csv`：执行结果原始序列化，保留动态 `metrics`、artifacts 和后端扩展；
- `report.jsonl` / `report.csv`：面向 CI、趋势系统和数据仓库的稳定字段投影。

稳定投影使用 `report_schema_version: 1`。新增字段只会以向后兼容方式添加；字段语义发生不兼容变化时必须提升 schema version。

## 核心字段域

- 身份与来源：`run_id`、`plan_id`、`case_id`、`case_name`、`case_fingerprint`、`matrix_id`、fuzz generation 信息；
- 执行：`task`、`status`、`node_name`、`backend`、`device_id`、`resolved_device`、`candidate_executor`；
- 精度：`comparison`、最大/平均绝对误差、最大/平均相对误差、`rmse`、`matched_ratio`；
- 性能与内存：`latency_ms`、`throughput_items_per_s`、`memory_peak_bytes`；
- 失败与环境：`failure_kind`、`failure_stage`、`error`、硬件 probe/fingerprint；
- 可扩展证据：`runtime_policy_json`、`metrics_json`、`evidence_json`。

复杂对象统一以 JSON 字符串放进 `*_json` 字段，因此 CSV 列顺序不受 backend、comparator 或 plugin 的动态 metrics 影响。JSONL 使用相同字段名和同一版本号。

每次 `cx run` 都会生成：

```text
<output>/results.jsonl     # 原始执行记录
<output>/results.csv       # 原始扁平记录
<output>/report.jsonl      # report_schema_version = 1
<output>/report.csv        # 与 report.jsonl 相同字段顺序
```

执行 `cx report` 后会额外生成 `<output>/report.md`，其中标注稳定导出路径与 schema 版本。
