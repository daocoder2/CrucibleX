import json
from pathlib import Path

from cruciblex.importers.msprof_summary import import_msprof_summary, write_msprof_summary


def _fixture(root: Path) -> Path:
    root.mkdir()
    (root / "op_statistic_1.csv").write_text("Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n0,Abs,AI_VECTOR_CORE,2,10.0,4.0,5.0,6.0,100\n", encoding="utf-8")
    (root / "task_time_1.csv").write_text("Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)\n0,kernel,AI_VECTOR_CORE,2,1,6.0,1,7\n", encoding="utf-8")
    (root / "op_summary_1.csv").write_text("Device_id,Op Name,OP Type,Task Duration(us),Task Wait Time(us),aiv_time(us)\n0,aclnnAbs,Abs,6.0,1.0,2.0\n", encoding="utf-8")
    return root


def test_import_msprof_summary_normalizes_exports(tmp_path):
    summary = import_msprof_summary(_fixture(tmp_path / "trace"))
    assert summary["status"] == "parsed"
    assert summary["device_count"] == 1
    assert summary["top_operators"][0]["total_time_us"] == 10.0
    assert summary["task_summary"][0]["total_time_us"] == 6.0
    output = write_msprof_summary(tmp_path / "trace", tmp_path / "summary.json")
    assert json.loads(output.read_text())["status"] == "parsed"


def test_import_msprof_summary_reports_missing_exports(tmp_path):
    summary = import_msprof_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["warnings"]
