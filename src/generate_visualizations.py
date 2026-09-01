"""将仿真CSV转换为可直接展示的HTML报告和SVG图表。"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "硬件配置",
    "模型",
    "模式",
    "input_seqlen",
    "history_seqlen",
    "tail_kv_tokens",
    "batchsize",
    "chip_num",
    "layer_num",
    "MFU (%)",
    "HBF read util (%)",
    "HBF write util (%)",
    "E2E latency(ms)",
]

DETAIL_FIELDS = [
    "层号",
    "layer_s(us)",
    "qkv_wload_s(us)",
    "qkv_wload_e(us)",
    "qkv_proj_s(us)",
    "qkv_proj_e(us)",
    "allgather_e(us)",
    "newkv_ddr_write_e(us)",
    "kv_hbf_load_s(us)",
    "kv_hbf_load_e(us)",
    "kv_ddr_load_s(us)",
    "kv_ddr_load_e(us)",
    "FA_s(us)",
    "FA_e(us)",
    "o_wload_s(us)",
    "o_wload_e(us)",
    "o_proj_s(us)",
    "o_proj_e(us)",
    "allreduce_e(us)",
    "ffn_wload_s(us)",
    "ffn_wload_e(us)",
    "ffn_s(us)",
    "ffn_e(us)",
    "scatterreduce_e(us)",
    "kv_write_s(us)",
    "kv_write_e(us)",
    "layer_e(us)",
]

METRICS = [
    ("MFU (%)", "MFU", "%", "#2563eb"),
    ("HBF read util (%)", "HBF读利用率", "%", "#0f766e"),
    ("HBF write util (%)", "HBF写利用率", "%", "#d97706"),
    ("E2E latency(ms)", "E2E延迟", "ms", "#be123c"),
]

RESOURCE_COLORS = {
    "HBF读": "#2563eb",
    "DDR": "#15803d",
    "NPU矩阵": "#d97706",
    "PCIe": "#7c3aed",
    "HBF写": "#be123c",
}

FONT_FAMILY = "Microsoft YaHei, PingFang SC, Arial, sans-serif"


class VisualizationInputError(ValueError):
    """输入CSV缺失或格式不正确。"""


def _read_csv(path: Path, required_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise VisualizationInputError(f"找不到CSV文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise VisualizationInputError(
                f"{path.name}缺少字段：{', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise VisualizationInputError(f"{path.name}没有数据行。")
    return rows


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualizationInputError(f"字段{field}不是有效数值。") from exc
    if not math.isfinite(value):
        raise VisualizationInputError(f"字段{field}必须是有限数值。")
    return value


def read_summary(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, SUMMARY_FIELDS)
    numeric_fields = SUMMARY_FIELDS[3:]
    for row in rows:
        for field in numeric_fields:
            value = _number(row, field)
            if value < 0:
                raise VisualizationInputError(f"字段{field}不能为负数。")
    return rows


def read_detail(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, DETAIL_FIELDS)
    for row in rows:
        for field in DETAIL_FIELDS:
            value = _number(row, field)
            if value < 0:
                raise VisualizationInputError(f"字段{field}不能为负数。")
    rows.sort(key=lambda row: _number(row, "层号"))
    return rows


def _scenario_label(row: dict[str, str]) -> str:
    return (
        f"{row['模式']}  B{row['batchsize']}  "
        f"I{row['input_seqlen']}  H{row['history_seqlen']}"
    )


def _svg_open(width: int, height: int, title: str, description: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="title desc">'
        f'<title id="title">{html.escape(title)}</title>'
        f'<desc id="desc">{html.escape(description)}</desc>'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
    )


def _nice_axis_max(value: float, minimum: float = 0.0) -> float:
    value = max(value, minimum)
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    step = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    return step * magnitude


def render_metric_comparison(rows: list[dict[str, str]]) -> str:
    row_height = 32
    panel_height = 76 + row_height * len(rows)
    width = 1200
    height = 44 + panel_height * len(METRICS)
    label_x = 228
    plot_x = 250
    plot_width = 830
    svg = [
        _svg_open(
            width,
            height,
            "仿真场景指标对比",
            "对比各场景的MFU、HBF读写利用率和端到端延迟。",
        ),
        f'<text x="32" y="32" font-family="{FONT_FAMILY}" font-size="20" '
        'font-weight="600" fill="#17202a">场景指标对比</text>',
    ]

    for metric_index, (field, label, unit, color) in enumerate(METRICS):
        panel_y = 44 + metric_index * panel_height
        values = [_number(row, field) for row in rows]
        minimum = 100.0 if unit == "%" else 0.0
        axis_max = _nice_axis_max(max(values) * 1.08, minimum)
        svg.append(
            f'<text x="32" y="{panel_y + 28}" font-family="{FONT_FAMILY}" '
            f'font-size="16" font-weight="600" fill="#17202a">{label} ({unit})</text>'
        )
        svg.append(
            f'<line x1="{plot_x}" y1="{panel_y + 45}" x2="{plot_x + plot_width}" '
            f'y2="{panel_y + 45}" stroke="#d5dadd" stroke-width="1"/>'
        )
        for tick in range(6):
            tick_value = axis_max * tick / 5
            tick_x = plot_x + plot_width * tick / 5
            svg.extend(
                [
                    f'<line x1="{tick_x:.1f}" y1="{panel_y + 42}" '
                    f'x2="{tick_x:.1f}" y2="{panel_y + panel_height - 10}" '
                    'stroke="#e7eaec" stroke-width="1"/>',
                    f'<text x="{tick_x:.1f}" y="{panel_y + 40}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="11" fill="#65717b">'
                    f'{tick_value:.0f}</text>',
                ]
            )
        if unit == "%" and axis_max > 100:
            ceiling_x = plot_x + plot_width * 100 / axis_max
            svg.append(
                f'<line x1="{ceiling_x:.1f}" y1="{panel_y + 42}" '
                f'x2="{ceiling_x:.1f}" y2="{panel_y + panel_height - 10}" '
                'stroke="#7b8790" stroke-width="1.5" stroke-dasharray="5 4"/>'
            )
        for row_index, (row, value) in enumerate(zip(rows, values)):
            y = panel_y + 58 + row_index * row_height
            bar_width = max(1.0, plot_width * value / axis_max)
            svg.extend(
                [
                    f'<text x="{label_x}" y="{y + 15}" text-anchor="end" '
                    f'font-family="{FONT_FAMILY}" font-size="12" fill="#34414b">'
                    f'{html.escape(_scenario_label(row))}</text>',
                    f'<rect x="{plot_x}" y="{y}" width="{bar_width:.1f}" height="20" '
                    f'rx="3" fill="{color}" fill-opacity="0.86"/>',
                    f'<text x="{min(plot_x + bar_width + 8, width - 70):.1f}" '
                    f'y="{y + 15}" font-family="{FONT_FAMILY}" font-size="12" '
                    f'fill="#17202a">{value:.3f}</text>',
                ]
            )
    svg.append("</svg>")
    return "".join(svg)


def _timeline_events(row: dict[str, str]) -> list[tuple[str, str, float, float]]:
    def event(
        label: str, resource: str, start_field: str, end_field: str
    ) -> tuple[str, str, float, float]:
        return label, resource, _number(row, start_field), _number(row, end_field)

    return [
        event("QKV权重读取", "HBF读", "qkv_wload_s(us)", "qkv_wload_e(us)"),
        event("Bulk KV读取", "HBF读", "kv_hbf_load_s(us)", "kv_hbf_load_e(us)"),
        event("O权重读取", "HBF读", "o_wload_s(us)", "o_wload_e(us)"),
        event("FFN权重读取", "HBF读", "ffn_wload_s(us)", "ffn_wload_e(us)"),
        event("New KV写DDR", "DDR", "qkv_proj_e(us)", "newkv_ddr_write_e(us)"),
        event("New/Tail KV读DDR", "DDR", "kv_ddr_load_s(us)", "kv_ddr_load_e(us)"),
        event("QKV投影", "NPU矩阵", "qkv_proj_s(us)", "qkv_proj_e(us)"),
        event("Flash Attention", "NPU矩阵", "FA_s(us)", "FA_e(us)"),
        event("O投影", "NPU矩阵", "o_proj_s(us)", "o_proj_e(us)"),
        event("FFN", "NPU矩阵", "ffn_s(us)", "ffn_e(us)"),
        event("Allgather", "PCIe", "qkv_proj_e(us)", "allgather_e(us)"),
        event("Allreduce", "PCIe", "o_proj_e(us)", "allreduce_e(us)"),
        event("Scatter-reduce", "PCIe", "ffn_e(us)", "scatterreduce_e(us)"),
        event("KV写HBF", "HBF写", "kv_write_s(us)", "kv_write_e(us)"),
    ]


def render_first_layer_timeline(rows: list[dict[str, str]]) -> str:
    row = rows[0]
    events = _timeline_events(row)
    layer_start = _number(row, "layer_s(us)")
    layer_end = _number(row, "layer_e(us)")
    span = max(layer_end - layer_start, 1e-9)
    width = 1280
    row_height = 31
    height = 126 + row_height * len(events)
    plot_x = 235
    plot_width = 810
    duration_x = 1070
    svg = [
        _svg_open(
            width,
            height,
            "首层流水时序",
            "展示首层HBF、DDR、NPU矩阵和PCIe任务的开始与结束时间。",
        ),
        f'<text x="32" y="32" font-family="{FONT_FAMILY}" font-size="20" '
        'font-weight="600" fill="#17202a">首层流水时序</text>',
    ]

    legend_x = 32
    for resource, color in RESOURCE_COLORS.items():
        svg.extend(
            [
                f'<rect x="{legend_x}" y="50" width="12" height="12" rx="2" '
                f'fill="{color}"/>',
                f'<text x="{legend_x + 18}" y="61" font-family="{FONT_FAMILY}" '
                f'font-size="12" fill="#34414b">{resource}</text>',
            ]
        )
        legend_x += 104

    axis_y = 88
    for tick in range(6):
        value = layer_start + span * tick / 5
        x = plot_x + plot_width * tick / 5
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" '
                f'y2="{height - 22}" stroke="#e2e6e9" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{axis_y - 8}" text-anchor="middle" '
                f'font-family="{FONT_FAMILY}" font-size="11" fill="#65717b">'
                f'{value:.1f} us</text>',
            ]
        )
    svg.append(
        f'<text x="{duration_x}" y="{axis_y - 8}" font-family="{FONT_FAMILY}" '
        'font-size="11" fill="#65717b">持续时间</text>'
    )

    for index, (label, resource, start, end) in enumerate(events):
        if end < start:
            raise VisualizationInputError(f"{label}的结束时间早于开始时间。")
        y = 102 + index * row_height
        x = plot_x + plot_width * (start - layer_start) / span
        bar_width = max(2.0, plot_width * (end - start) / span)
        color = RESOURCE_COLORS[resource]
        svg.extend(
            [
                f'<text x="{plot_x - 14}" y="{y + 15}" text-anchor="end" '
                f'font-family="{FONT_FAMILY}" font-size="12" fill="#34414b">'
                f'{html.escape(label)}</text>',
                f'<line x1="{plot_x}" y1="{y + 20}" x2="{plot_x + plot_width}" '
                f'y2="{y + 20}" stroke="#f0f2f3" stroke-width="1"/>',
                f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="20" '
                f'rx="3" fill="{color}" fill-opacity="0.86"/>',
                f'<text x="{duration_x}" y="{y + 15}" font-family="{FONT_FAMILY}" '
                f'font-size="12" fill="#34414b">{end - start:.3f} us</text>',
            ]
        )
    svg.append("</svg>")
    return "".join(svg)


def render_cumulative_latency(rows: list[dict[str, str]]) -> str:
    points = [
        (_number(row, "层号"), _number(row, "layer_e(us)") / 1000)
        for row in rows
    ]
    width = 1200
    height = 460
    plot_x = 92
    plot_y = 60
    plot_width = 1010
    plot_height = 320
    min_layer = min(point[0] for point in points)
    max_layer = max(point[0] for point in points)
    max_latency = max(point[1] for point in points)
    x_span = max(max_layer - min_layer, 1)
    y_max = _nice_axis_max(max_latency * 1.08)

    def x_position(layer: float) -> float:
        return plot_x + plot_width * (layer - min_layer) / x_span

    def y_position(latency: float) -> float:
        return plot_y + plot_height * (1 - latency / y_max)

    polyline = " ".join(
        f"{x_position(layer):.1f},{y_position(latency):.1f}"
        for layer, latency in points
    )
    svg = [
        _svg_open(
            width,
            height,
            "逐层累计延迟",
            "展示各层结束时刻随层号累积的端到端延迟。",
        ),
        f'<text x="32" y="32" font-family="{FONT_FAMILY}" font-size="20" '
        'font-weight="600" fill="#17202a">逐层累计延迟</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        svg.extend(
            [
                f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" '
                f'y2="{y:.1f}" stroke="#e2e6e9" stroke-width="1"/>',
                f'<text x="{plot_x - 12}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="{FONT_FAMILY}" font-size="11" fill="#65717b">'
                f'{value:.2f}</text>',
            ]
        )
    x_ticks = sorted(
        {
            round(min_layer + x_span * tick / 6)
            for tick in range(7)
        }
    )
    for layer in x_ticks:
        x = x_position(layer)
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" '
                f'y2="{plot_y + plot_height}" stroke="#f0f2f3" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{plot_y + plot_height + 24}" '
                f'text-anchor="middle" font-family="{FONT_FAMILY}" font-size="11" '
                f'fill="#65717b">{layer}</text>',
            ]
        )
    svg.extend(
        [
            f'<polyline points="{polyline}" fill="none" stroke="#0f766e" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<circle cx="{x_position(points[-1][0]):.1f}" '
            f'cy="{y_position(points[-1][1]):.1f}" r="5" fill="#0f766e"/>',
            f'<text x="{x_position(points[-1][0]) - 8:.1f}" '
            f'y="{y_position(points[-1][1]) - 12:.1f}" text-anchor="end" '
            f'font-family="{FONT_FAMILY}" font-size="13" font-weight="600" '
            f'fill="#17202a">{points[-1][1]:.3f} ms</text>',
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{height - 24}" '
            f'text-anchor="middle" font-family="{FONT_FAMILY}" font-size="12" '
            'fill="#65717b">层号</text>',
            f'<text x="24" y="{plot_y + plot_height / 2:.1f}" '
            f'transform="rotate(-90 24 {plot_y + plot_height / 2:.1f})" '
            f'text-anchor="middle" font-family="{FONT_FAMILY}" font-size="12" '
            'fill="#65717b">累计延迟 (ms)</text>',
            "</svg>",
        ]
    )
    return "".join(svg)


def _summary_table(rows: list[dict[str, str]]) -> str:
    columns = [
        "模式",
        "input_seqlen",
        "history_seqlen",
        "batchsize",
        "MFU (%)",
        "HBF read util (%)",
        "HBF write util (%)",
        "E2E latency(ms)",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if column in {metric[0] for metric in METRICS}:
                value = f"{_number(row, column):.6f}"
            cells.append(f"<td>{html.escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_html_report(
    summary_rows: list[dict[str, str]],
    metric_svg: str,
    timeline_svg: str | None,
    cumulative_svg: str | None,
) -> str:
    first = summary_rows[0]
    kpis = ""
    if len(summary_rows) == 1:
        cards = []
        for field, label, unit, _ in METRICS:
            cards.append(
                '<div class="kpi">'
                f'<div class="kpi-label">{html.escape(label)}</div>'
                f'<div class="kpi-value">{_number(first, field):.3f}'
                f'<span>{html.escape(unit)}</span></div>'
                "</div>"
            )
        kpis = f'<section class="kpis">{"".join(cards)}</section>'

    detail_sections = ""
    if timeline_svg and cumulative_svg:
        detail_sections = (
            '<section><h2>首层资源时序</h2>'
            f'<div class="chart">{timeline_svg}</div></section>'
            '<section><h2>48层延迟累积</h2>'
            f'<div class="chart">{cumulative_svg}</div></section>'
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HBF MFU仿真结果</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #17202a;
      background: #f5f6f7;
      font-family: {FONT_FAMILY};
      letter-spacing: 0;
    }}
    header {{
      background: #ffffff;
      border-bottom: 4px solid #0f766e;
    }}
    .wrap {{ width: min(1280px, calc(100% - 40px)); margin: 0 auto; }}
    header .wrap {{ padding: 28px 0 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; font-weight: 600; }}
    h2 {{ margin: 0 0 18px; font-size: 20px; font-weight: 600; }}
    .meta {{ color: #59656f; font-size: 14px; }}
    .boundary {{
      display: inline-block;
      margin-top: 12px;
      padding: 5px 9px;
      border: 1px solid #d97706;
      border-radius: 4px;
      color: #8a4b08;
      background: #fff8e7;
      font-size: 13px;
    }}
    main {{ padding: 24px 0 48px; }}
    section {{
      padding: 24px 0;
      border-bottom: 1px solid #d9dee2;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding-top: 0;
    }}
    .kpi {{
      min-width: 0;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #d9dee2;
      border-radius: 6px;
    }}
    .kpi-label {{ color: #59656f; font-size: 13px; }}
    .kpi-value {{
      margin-top: 8px;
      font-size: 26px;
      font-weight: 600;
      overflow-wrap: anywhere;
    }}
    .kpi-value span {{
      margin-left: 5px;
      color: #59656f;
      font-size: 13px;
      font-weight: 400;
    }}
    .chart {{
      width: 100%;
      overflow: hidden;
      background: #ffffff;
    }}
    .chart svg {{ display: block; width: 100%; height: auto; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border: 1px solid #d9dee2;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f3; font-weight: 600; }}
    @media (max-width: 760px) {{
      .wrap {{ width: min(100% - 24px, 1280px); }}
      h1 {{ font-size: 24px; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 420px) {{
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>HBF MFU仿真结果</h1>
      <div class="meta">{html.escape(first['硬件配置'])} · {html.escape(first['模型'])} · {len(summary_rows)}个场景</div>
      <div class="boundary">公式模型输出，未进行真实硬件校准</div>
    </div>
  </header>
  <main class="wrap">
    {kpis}
    <section>
      <h2>核心指标</h2>
      <div class="chart">{metric_svg}</div>
    </section>
    {detail_sections}
    <section>
      <h2>场景数据</h2>
      {_summary_table(summary_rows)}
    </section>
  </main>
</body>
</html>
"""


def generate_report(
    summary_csv: Path, detail_csv: Path | None, output_dir: Path
) -> list[Path]:
    summary_rows = read_summary(summary_csv)
    detail_rows = read_detail(detail_csv) if detail_csv else None
    metric_svg = render_metric_comparison(summary_rows)
    timeline_svg = render_first_layer_timeline(detail_rows) if detail_rows else None
    cumulative_svg = render_cumulative_latency(detail_rows) if detail_rows else None

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    metric_path = output_dir / "场景指标对比.svg"
    metric_path.write_text(metric_svg, encoding="utf-8")
    outputs.append(metric_path)

    if timeline_svg and cumulative_svg:
        timeline_path = output_dir / "首层流水时序.svg"
        timeline_path.write_text(timeline_svg, encoding="utf-8")
        outputs.append(timeline_path)

        cumulative_path = output_dir / "48层累计延迟.svg"
        cumulative_path.write_text(cumulative_svg, encoding="utf-8")
        outputs.append(cumulative_path)

    report_path = output_dir / "仿真结果总览.html"
    report_path.write_text(
        render_html_report(
            summary_rows, metric_svg, timeline_svg, cumulative_svg
        ),
        encoding="utf-8",
    )
    outputs.append(report_path)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从仿真CSV生成HTML与SVG可视化。")
    parser.add_argument("--summary-csv", type=Path, required=True, help="仿真汇总CSV")
    parser.add_argument("--detail-csv", type=Path, help="可选的逐层时序CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="可视化输出目录")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = generate_report(args.summary_csv, args.detail_csv, args.output_dir)
    except VisualizationInputError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2
    print("可视化生成完成")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
