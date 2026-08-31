# 运行版本与输入 Provenance

每个 CrucibleX run 都必须能说明生成了什么输入、由哪个 Driver 生成、以及执行 worker 是否与 Driver 使用同一 pipeline。该契约不替代硬件 evidence，而是为结果的可复核性提供运行与输入上下文。

## 输入 provenance

`inputs` artifact metadata 使用 `input_schema_version: 1`，并包含 `case_fingerprint`、生成器、seed，以及每个参数的来源和参数声明 fingerprint。来源可以是 `exact_values`、`value_range`、`value_policy` 或 `default`。输入数据仍保持原有可重放 JSON 结构，控制信息放在 artifact metadata。

## Driver 与 worker 版本

resource discovery 的 runtime probe 会记录 CrucibleX 版本和 `pipeline_sha256`。Driver 对已探测 worker 计算兼容性：`matched` 表示全部相同，`mismatched` 表示至少一个不同，`unavailable` 表示没有足够 probe。

`cx run --version-policy warn` 是默认行为：继续执行，并将 compatibility 结论写入 discovery、manifest 和报告。`--version-policy strict` 仅在确证 `mismatched` 时于提交任务前拒绝执行，同时保留 discovery snapshot。

旧 worker 的结果可以由 Driver 从已有 GPU probe metrics 和 artifact 归一化 evidence，但 manifest 仍必须如实记录 pipeline 是否一致。
