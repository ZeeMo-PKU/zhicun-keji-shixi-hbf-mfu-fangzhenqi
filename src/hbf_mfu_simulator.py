#!/usr/bin/env python3
"""M12-24B HBF MFU simulator with minimal corrections to the source formula."""

from __future__ import annotations

import argparse
import csv
import math
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_NAME = "M12-24B"
LAYER_COUNT = 48
HIDDEN_SIZE = 5120
QKV_OUTPUT_SIZE = 8192
O_PROJ_INPUT_SIZE = 6144
FFN_INTERMEDIATE_SIZE = 1536
TOP_K = 8
EXPERT_COUNT = 384
Q_HEADS = 48
KV_HEADS = 8
HEAD_DIM = 128
ELEMENT_BYTES = 1  # W8A8 and KV int8
KV_WRITE_MIN_SECONDS = 250e-6
VALID_CHIP_NUMS = {1, 2, 4, 8}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class InputError(ValueError):
    """Raised when an input file or scenario is incomplete or inconsistent."""


@dataclass(frozen=True)
class Hardware:
    config_name: str
    chip_num: int
    mat_flops: float
    util: float
    hbf_bw: float
    hbf_w_bw: float
    hbf_lat: float
    ddr_bw: float
    pcie_bw: float
    pcie_lat: float


@dataclass(frozen=True)
class Scenario:
    mode: str
    batch_size: int
    input_seqlen: int
    history_seqlen: int
    tail_kv_tokens: int


@dataclass(frozen=True)
class Workload:
    p_qkv_w: int
    p_qkv_proj: int
    p_bulk_kv: int
    p_tail_kv: int
    p_newkv: int
    p_o: int
    p_attn_out: int
    p_topk_ffn: int
    p_ffn_out: int
    c_qkv_proj: int
    c_fa: int
    c_o_proj: int
    c_ffn: int

    @property
    def matrix_ops(self) -> int:
        return self.c_qkv_proj + self.c_fa + self.c_o_proj + self.c_ffn

    @property
    def hbf_read_bytes(self) -> int:
        return self.p_qkv_w + self.p_bulk_kv + self.p_o + self.p_topk_ffn

    @property
    def hbf_write_bytes(self) -> int:
        return self.p_tail_kv + self.p_newkv


def matrix_multiply_ops(m: int, k: int, n: int) -> int:
    """Count multiplications and additions for [m,k] x [k,n]."""
    if min(m, k, n) <= 0:
        raise InputError("矩阵维度必须为正整数。")
    return m * n * (2 * k - 1)


def _column_name(cell_ref: str) -> str:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise InputError(f"无法识别Excel单元格地址：{cell_ref}")
    return match.group(1)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return values


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") != sheet_name:
            continue
        relation_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
        if relation_id not in relation_targets:
            break
        target = relation_targets[relation_id].lstrip("/")
        return target if target.startswith("xl/") else posixpath.normpath(f"xl/{target}")
    raise InputError(f"硬件文件中缺少工作表：{sheet_name}")


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
        return text or None
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def read_hardware_excel(path: Path) -> Hardware:
    if not path.is_file():
        raise InputError(f"硬件文件不存在：{path}")
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            root = ET.fromstring(archive.read(_sheet_path(archive, "硬件指标")))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise InputError(f"无法读取硬件Excel文件：{exc}") from exc

    rows: list[dict[str, str | None]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
        values: dict[str, str | None] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            values[_column_name(cell.attrib["r"])] = _cell_value(cell, shared_strings)
        rows.append(values)

    value_headers = {"用户填写值", "填写值"}
    header = next(
        (
            row
            for row in rows
            if "公式变量" in row.values()
            and value_headers.intersection(row.values())
            and "单位" in row.values()
        ),
        None,
    )
    if header is None:
        raise InputError("硬件指标表缺少标准表头。")
    variable_col = next(key for key, value in header.items() if value == "公式变量")
    value_col = next(key for key, value in header.items() if value in value_headers)
    unit_col = next(key for key, value in header.items() if value == "单位")

    entries: dict[str, tuple[str | None, str | None]] = {}
    passed_header = False
    for row in rows:
        if row is header:
            passed_header = True
            continue
        if not passed_header:
            continue
        variable = row.get(variable_col)
        if variable:
            entries[variable.strip()] = (row.get(value_col), row.get(unit_col))

    expected_units = {
        "config_name": "-",
        "chip_num": "个",
        "mat_flops": "TOPS",
        "util": "比例",
        "hbf_bw": "GB/s",
        "hbf_w_bw": "MB/s",
        "hbf_lat": "us",
        "ddr_bw": "GB/s",
        "pcie_bw": "GB/s",
        "pcie_lat": "us",
    }
    missing = [
        key
        for key in expected_units
        if key not in entries or entries[key][0] is None or not entries[key][0].strip()
    ]
    if missing:
        raise InputError("硬件指标未填写完整，缺少：" + ", ".join(missing))
    for key, expected_unit in expected_units.items():
        actual_unit = (entries[key][1] or "").strip()
        if actual_unit != expected_unit:
            raise InputError(
                f"{key}的单位应为{expected_unit}，实际为{actual_unit or '空'}。"
            )

    def number(key: str) -> float:
        raw = entries[key][0]
        assert raw is not None
        try:
            value = float(raw)
        except ValueError as exc:
            raise InputError(f"{key}必须填写数值，实际为：{raw}") from exc
        if not math.isfinite(value):
            raise InputError(f"{key}必须是有限数值。")
        return value

    chip_value = number("chip_num")
    if not chip_value.is_integer():
        raise InputError("chip_num必须是整数。")
    chip_num = int(chip_value)
    if chip_num not in VALID_CHIP_NUMS:
        raise InputError("当前固定模型只支持chip_num为1、2、4或8。")

    util = number("util")
    if not 0 < util <= 1:
        raise InputError("util必须大于0且不超过1。")
    positive_keys = [
        "mat_flops",
        "hbf_bw",
        "hbf_w_bw",
        "hbf_lat",
        "ddr_bw",
        "pcie_bw",
        "pcie_lat",
    ]
    positive_values = {key: number(key) for key in positive_keys}
    non_positive = [key for key, value in positive_values.items() if value <= 0]
    if non_positive:
        raise InputError("以下硬件指标必须为正数：" + ", ".join(non_positive))

    config_name = entries["config_name"][0]
    assert config_name is not None
    return Hardware(
        config_name=config_name.strip(),
        chip_num=chip_num,
        mat_flops=positive_values["mat_flops"] * 1e12,
        util=util,
        hbf_bw=positive_values["hbf_bw"] * 1e9,
        hbf_w_bw=positive_values["hbf_w_bw"] * 1e6,
        hbf_lat=positive_values["hbf_lat"] * 1e-6,
        ddr_bw=positive_values["ddr_bw"] * 1e9,
        pcie_bw=positive_values["pcie_bw"] * 1e9,
        pcie_lat=positive_values["pcie_lat"] * 1e-6,
    )


def validate_scenario(scenario: Scenario) -> None:
    if scenario.mode not in {"prefill", "decode"}:
        raise InputError("mode必须为prefill或decode。")
    if scenario.batch_size <= 0 or scenario.input_seqlen <= 0:
        raise InputError("batch_size和input_seqlen必须为正整数。")
    if scenario.history_seqlen < 0:
        raise InputError("history_seqlen不能为负数。")
    if not 0 <= scenario.tail_kv_tokens <= scenario.history_seqlen:
        raise InputError("tail_kv_tokens必须位于0和history_seqlen之间。")


def build_workload(hardware: Hardware, scenario: Scenario) -> Workload:
    validate_scenario(scenario)
    n = hardware.chip_num
    for name, value in {
        "QKV输出维度": QKV_OUTPUT_SIZE,
        "O投影输入维度": O_PROJ_INPUT_SIZE,
        "Q head数量": Q_HEADS,
        "KV head数量": KV_HEADS,
        "top-k": TOP_K,
    }.items():
        if value % n:
            raise InputError(f"{name}={value}不能被chip_num={n}整除。")

    batch = scenario.batch_size
    seq = scenario.input_seqlen
    history = scenario.history_seqlen
    tail = scenario.tail_kv_tokens
    query_tokens = batch * seq
    kv_heads_per_chip = KV_HEADS // n
    q_heads_per_chip = Q_HEADS // n
    experts_per_chip = TOP_K // n
    bulk_tokens = history - tail

    pairs_per_request = seq * history + seq * (seq + 1) // 2
    qk_ops = batch * q_heads_per_chip * pairs_per_request * (2 * HEAD_DIM - 1)
    pv_ops = (
        batch
        * q_heads_per_chip
        * HEAD_DIM
        * (2 * pairs_per_request - seq)
    )

    return Workload(
        p_qkv_w=HIDDEN_SIZE * (QKV_OUTPUT_SIZE // n) * ELEMENT_BYTES,
        p_qkv_proj=query_tokens * QKV_OUTPUT_SIZE * ELEMENT_BYTES,
        p_bulk_kv=(
            batch * bulk_tokens * 2 * kv_heads_per_chip * HEAD_DIM * ELEMENT_BYTES
        ),
        p_tail_kv=(
            batch * tail * 2 * kv_heads_per_chip * HEAD_DIM * ELEMENT_BYTES
        ),
        p_newkv=(
            batch * seq * 2 * kv_heads_per_chip * HEAD_DIM * ELEMENT_BYTES
        ),
        p_o=(O_PROJ_INPUT_SIZE // n) * HIDDEN_SIZE * ELEMENT_BYTES,
        p_attn_out=query_tokens * HIDDEN_SIZE * ELEMENT_BYTES,
        p_topk_ffn=(
            batch
            * experts_per_chip
            * HIDDEN_SIZE
            * FFN_INTERMEDIATE_SIZE
            * 3
            * ELEMENT_BYTES
        ),
        p_ffn_out=query_tokens * HIDDEN_SIZE * ELEMENT_BYTES,
        c_qkv_proj=matrix_multiply_ops(
            query_tokens, HIDDEN_SIZE, QKV_OUTPUT_SIZE // n
        ),
        c_fa=qk_ops + pv_ops,
        c_o_proj=matrix_multiply_ops(
            query_tokens, O_PROJ_INPUT_SIZE // n, HIDDEN_SIZE
        ),
        c_ffn=experts_per_chip
        * (
            2
            * matrix_multiply_ops(
                query_tokens, HIDDEN_SIZE, FFN_INTERMEDIATE_SIZE
            )
            + matrix_multiply_ops(
                query_tokens, FFN_INTERMEDIATE_SIZE, HIDDEN_SIZE
            )
        ),
    )


def _hbf_read_end(start: float, size_bytes: int, hardware: Hardware) -> tuple[float, float]:
    if size_bytes == 0:
        return start, start
    first_ready = start + hardware.hbf_lat
    return first_ready, first_ready + size_bytes / hardware.hbf_bw


def simulate_layer(
    layer_s: float, hardware: Hardware, workload: Workload
) -> dict[str, float]:
    qkv_wload_s = layer_s
    qkv_first_ready, qkv_wload_e = _hbf_read_end(
        qkv_wload_s, workload.p_qkv_w, hardware
    )
    qkv_proj_s = qkv_first_ready
    qkv_proj_e = max(
        qkv_wload_e,
        qkv_proj_s + workload.c_qkv_proj / hardware.mat_flops / hardware.util,
    )

    allgather_e = (
        qkv_proj_e
        + hardware.pcie_lat
        + workload.p_qkv_proj / hardware.chip_num / hardware.pcie_bw
    )

    kv_hbf_load_s = qkv_wload_e
    kv_hbf_first_ready, kv_hbf_load_e = _hbf_read_end(
        kv_hbf_load_s, workload.p_bulk_kv, hardware
    )

    # Minimal flow correction: current KV is written to DDR before DDR reads
    # current KV together with the pre-existing tail KV.
    newkv_ddr_write_e = qkv_proj_e + workload.p_newkv / hardware.ddr_bw
    kv_ddr_load_s = max(allgather_e, newkv_ddr_write_e)
    kv_ddr_load_e = (
        kv_ddr_load_s
        + (workload.p_newkv + workload.p_tail_kv) / hardware.ddr_bw
    )

    fa_s = max(allgather_e, kv_hbf_first_ready, kv_ddr_load_s)
    fa_e = max(
        fa_s + workload.c_fa / hardware.mat_flops / hardware.util,
        kv_ddr_load_e,
        kv_hbf_load_e,
    )

    o_wload_s = kv_hbf_load_e
    o_first_ready, o_wload_e = _hbf_read_end(o_wload_s, workload.p_o, hardware)
    o_proj_s = max(fa_e, o_first_ready)
    o_proj_e = max(
        o_proj_s + workload.c_o_proj / hardware.mat_flops / hardware.util,
        o_wload_e,
    )

    allreduce_e = (
        o_proj_e
        + hardware.pcie_lat
        + 2 * workload.p_attn_out / hardware.pcie_bw
    )

    ffn_wload_s = allreduce_e
    ffn_first_ready, ffn_wload_e = _hbf_read_end(
        ffn_wload_s, workload.p_topk_ffn, hardware
    )
    ffn_s = ffn_first_ready
    ffn_e = max(
        ffn_s + workload.c_ffn / hardware.mat_flops / hardware.util,
        ffn_wload_e,
    )

    # Source correction: FFN output is a byte payload, not an operation count.
    scatterreduce_e = (
        ffn_e + hardware.pcie_lat + workload.p_ffn_out / hardware.pcie_bw
    )

    kv_write_s = ffn_wload_e
    kv_write_duration = max(
        hardware.hbf_lat
        + (workload.p_newkv + workload.p_tail_kv) / hardware.hbf_w_bw,
        KV_WRITE_MIN_SECONDS,
    )
    kv_write_e = kv_write_s + kv_write_duration
    layer_e = max(scatterreduce_e, kv_write_e)

    return {
        "layer_s": layer_s,
        "qkv_wload_s": qkv_wload_s,
        "qkv_wload_e": qkv_wload_e,
        "qkv_proj_s": qkv_proj_s,
        "qkv_proj_e": qkv_proj_e,
        "allgather_e": allgather_e,
        "newkv_ddr_write_e": newkv_ddr_write_e,
        "kv_hbf_load_s": kv_hbf_load_s,
        "kv_hbf_load_e": kv_hbf_load_e,
        "kv_ddr_load_s": kv_ddr_load_s,
        "kv_ddr_load_e": kv_ddr_load_e,
        "FA_s": fa_s,
        "FA_e": fa_e,
        "o_wload_s": o_wload_s,
        "o_wload_e": o_wload_e,
        "o_proj_s": o_proj_s,
        "o_proj_e": o_proj_e,
        "allreduce_e": allreduce_e,
        "ffn_wload_s": ffn_wload_s,
        "ffn_wload_e": ffn_wload_e,
        "ffn_s": ffn_s,
        "ffn_e": ffn_e,
        "scatterreduce_e": scatterreduce_e,
        "kv_write_s": kv_write_s,
        "kv_write_e": kv_write_e,
        "layer_e": layer_e,
    }


def simulate(
    hardware: Hardware, scenario: Scenario
) -> tuple[dict[str, Any], list[dict[str, float]], Workload]:
    workload = build_workload(hardware, scenario)
    traces: list[dict[str, float]] = []
    layer_s = 0.0
    for _ in range(LAYER_COUNT):
        trace = simulate_layer(layer_s, hardware, workload)
        traces.append(trace)
        layer_s = trace["layer_e"]

    e2e_seconds = traces[-1]["layer_e"]
    total_ops = workload.matrix_ops * LAYER_COUNT
    total_read_bytes = workload.hbf_read_bytes * LAYER_COUNT
    total_write_bytes = workload.hbf_write_bytes * LAYER_COUNT
    summary: dict[str, Any] = {
        "硬件配置": hardware.config_name,
        "模型": MODEL_NAME,
        "模式": scenario.mode,
        "input_seqlen": scenario.input_seqlen,
        "history_seqlen": scenario.history_seqlen,
        "tail_kv_tokens": scenario.tail_kv_tokens,
        "batchsize": scenario.batch_size,
        "chip_num": hardware.chip_num,
        "layer_num": LAYER_COUNT,
        "MFU (%)": 100 * total_ops / (hardware.mat_flops * e2e_seconds),
        "HBF read util (%)": 100
        * total_read_bytes
        / (hardware.hbf_bw * e2e_seconds),
        "HBF write util (%)": 100
        * total_write_bytes
        / (hardware.hbf_w_bw * e2e_seconds),
        "E2E latency(ms)": 1000 * e2e_seconds,
    }
    return summary, traces, workload


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def write_detail_csv(path: Path, traces: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timing_fields = list(traces[0])
    fieldnames = ["层号"] + [f"{name}(us)" for name in timing_fields]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_index, trace in enumerate(traces, start=1):
            row: dict[str, Any] = {"层号": layer_index}
            row.update({f"{name}(us)": value * 1e6 for name, value in trace.items()})
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取硬件指标Excel，并按最小修正后的文档公式仿真M12-24B。"
    )
    parser.add_argument("--hardware-file", type=Path, required=True, help="硬件指标Excel")
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-seqlen", type=int, required=True)
    parser.add_argument("--history-seqlen", type=int, required=True)
    parser.add_argument(
        "--tail-kv-tokens",
        type=int,
        help="DDR中尚未迁移到HBF的历史KV长度；history为0时可省略",
    )
    parser.add_argument("--csv", type=Path, required=True, help="汇总结果CSV")
    parser.add_argument("--detail-csv", type=Path, help="可选的48层逐步时间CSV")
    parser.add_argument(
        "--visual-dir", type=Path, help="可选的HTML与SVG可视化输出目录"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    visual_outputs: list[Path] = []
    try:
        tail_kv_tokens = args.tail_kv_tokens
        if tail_kv_tokens is None:
            if args.history_seqlen == 0:
                tail_kv_tokens = 0
            else:
                raise InputError(
                    "history_seqlen大于0时必须提供--tail-kv-tokens；"
                    "任务书没有给出独立decode场景的tail长度。"
                )
        scenario = Scenario(
            mode=args.mode,
            batch_size=args.batch_size,
            input_seqlen=args.input_seqlen,
            history_seqlen=args.history_seqlen,
            tail_kv_tokens=tail_kv_tokens,
        )
        hardware = read_hardware_excel(args.hardware_file)
        summary, traces, _ = simulate(hardware, scenario)
        write_summary_csv(args.csv, summary)
        if args.detail_csv:
            write_detail_csv(args.detail_csv, traces)
        if args.visual_dir:
            from src.generate_visualizations import generate_report

            visual_outputs = generate_report(
                args.csv, args.detail_csv, args.visual_dir
            )
    except InputError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2

    print("仿真完成")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.9f}")
        else:
            print(f"{key}: {value}")
    print(f"汇总结果：{args.csv}")
    if args.detail_csv:
        print(f"逐层明细：{args.detail_csv}")
    for path in visual_outputs:
        print(f"可视化：{path}")
    print("AFD指标未输出：任务书仍缺少AF分离时序公式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
