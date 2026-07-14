"""可视化生成器测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from 生成仿真可视化 import (
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    VisualizationInputError,
    generate_report,
)


SUMMARY_ROW = {
    "硬件配置": "测试硬件",
    "模型": "M12-24B",
    "模式": "prefill",
    "input_seqlen": "1024",
    "history_seqlen": "0",
    "tail_kv_tokens": "0",
    "batchsize": "1",
    "chip_num": "8",
    "layer_num": "48",
    "MFU (%)": "45.9",
    "HBF read util (%)": "5.3",
    "HBF write util (%)": "36.5",
    "E2E latency(ms)": "213.0",
}

FIRST_LAYER = {
    "层号": "1",
    "layer_s(us)": "0",
    "qkv_wload_s(us)": "0",
    "qkv_wload_e(us)": "10",
    "qkv_proj_s(us)": "1",
    "qkv_proj_e(us)": "12",
    "allgather_e(us)": "14",
    "newkv_ddr_write_e(us)": "13",
    "kv_hbf_load_s(us)": "10",
    "kv_hbf_load_e(us)": "25",
    "kv_ddr_load_s(us)": "14",
    "kv_ddr_load_e(us)": "18",
    "FA_s(us)": "14",
    "FA_e(us)": "26",
    "o_wload_s(us)": "25",
    "o_wload_e(us)": "35",
    "o_proj_s(us)": "26",
    "o_proj_e(us)": "37",
    "allreduce_e(us)": "40",
    "ffn_wload_s(us)": "40",
    "ffn_wload_e(us)": "65",
    "ffn_s(us)": "41",
    "ffn_e(us)": "70",
    "scatterreduce_e(us)": "74",
    "kv_write_s(us)": "65",
    "kv_write_e(us)": "75",
    "layer_e(us)": "75",
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class VisualizationTests(unittest.TestCase):
    def test_generates_html_and_three_svg_charts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary.csv"
            detail_path = root / "detail.csv"
            output_dir = root / "visual"
            second_layer = {"层号": "2"}
            for field in DETAIL_FIELDS[1:]:
                second_layer[field] = str(float(FIRST_LAYER[field]) + 75)

            _write_csv(summary_path, SUMMARY_FIELDS, [SUMMARY_ROW])
            _write_csv(
                detail_path, DETAIL_FIELDS, [FIRST_LAYER, second_layer]
            )
            outputs = generate_report(summary_path, detail_path, output_dir)

            self.assertEqual(len(outputs), 4)
            self.assertTrue((output_dir / "仿真结果总览.html").is_file())
            self.assertIn(
                "公式模型输出，未进行真实硬件校准",
                (output_dir / "仿真结果总览.html").read_text(encoding="utf-8"),
            )
            for filename in [
                "场景指标对比.svg",
                "首层流水时序.svg",
                "48层累计延迟.svg",
            ]:
                ET.fromstring((output_dir / filename).read_text(encoding="utf-8"))

    def test_rejects_summary_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary.csv"
            fields = [field for field in SUMMARY_FIELDS if field != "MFU (%)"]
            row = {field: SUMMARY_ROW[field] for field in fields}
            _write_csv(summary_path, fields, [row])

            with self.assertRaisesRegex(VisualizationInputError, "MFU"):
                generate_report(summary_path, None, root / "visual")


if __name__ == "__main__":
    unittest.main()
