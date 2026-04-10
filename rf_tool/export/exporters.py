"""
Export utilities for RF System Tool.

Provides:
  - CSV export of cascade metrics
  - PNG/PDF image export of the canvas view
  - HTML report generation
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Any, Optional


# ======================================================================= #
# CSV Export                                                               #
# ======================================================================= #

def export_cascade_csv(
    metrics: Dict,
    blocks: list,
    filepath: str = "cascade_metrics.csv",
) -> None:
    """
    Export per-stage and total cascade metrics to a CSV file.

    Parameters
    ----------
    metrics : dict
        Output of compute_cascade_metrics().
    blocks : list of RFBlock
    filepath : str
    """
    rows = []
    stage_gains = metrics.get("stage_gains", [])
    stage_nfs = metrics.get("stage_nfs", [])
    cum_gains = metrics.get("cumulative_gains", [])

    for i, block in enumerate(blocks):
        rows.append({
            "Stage": i + 1,
            "Block Type": block.BLOCK_TYPE,
            "Label": block.label,
            "Gain (dB)": f"{stage_gains[i]:.3f}" if i < len(stage_gains) else "",
            "NF (dB)": f"{stage_nfs[i]:.3f}" if i < len(stage_nfs) else "",
            "Cumulative Gain (dB)": f"{cum_gains[i]:.3f}" if i < len(cum_gains) else "",
            "P1dB out (dBm)": f"{block.p1db_dbm:.1f}" if block.p1db_dbm is not None else "N/A",
            "OIP3 (dBm)": f"{block.oip3_dbm:.1f}" if block.oip3_dbm is not None else "N/A",
            "Max Input (dBm)": (f"{block.max_input_power_dbm:.1f}"
                                if block.max_input_power_dbm is not None else "N/A"),
        })

    # Summary row
    iip3 = metrics.get("iip3_dbm")
    oip3 = metrics.get("oip3_dbm")
    p1db = metrics.get("p1db_in_dbm")
    rows.append({
        "Stage": "TOTAL",
        "Block Type": "",
        "Label": "",
        "Gain (dB)": f"{metrics.get('gain_db', 0):.3f}",
        "NF (dB)": f"{metrics.get('nf_db', 0):.3f}",
        "Cumulative Gain (dB)": "",
        "P1dB out (dBm)": f"{p1db:.1f}" if p1db is not None else "N/A",
        "OIP3 (dBm)": f"{oip3:.1f}" if oip3 is not None else "N/A",
        "Max Input (dBm)": (f"{metrics.get('min_damage_dbm'):.1f}"
                            if metrics.get("min_damage_dbm") is not None else "N/A"),
    })

    fieldnames = [
        "Stage", "Block Type", "Label", "Gain (dB)", "NF (dB)",
        "Cumulative Gain (dB)", "P1dB out (dBm)", "OIP3 (dBm)", "Max Input (dBm)",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================= #
# Image Export (canvas)                                                    #
# ======================================================================= #

def export_canvas_image(scene, filepath: str = "canvas.png") -> None:
    """
    Export the QGraphicsScene to a PNG or PDF file.

    Parameters
    ----------
    scene : QGraphicsScene
    filepath : str
        .png or .pdf extension determines format.
    """
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPainter, QPixmap, QImage
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtPrintSupport import QPrinter

    rect = scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filepath)
        printer.setPageSize(QPrinter.A4)
        painter = QPainter(printer)
        scene.render(painter, source=rect)
        painter.end()
    else:
        # PNG
        img = QImage(int(rect.width()) + 4, int(rect.height()) + 4,
                     QImage.Format_ARGB32)
        img.fill(Qt.white)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        scene.render(painter, source=rect)
        painter.end()
        img.save(filepath)


# ======================================================================= #
# HTML Report                                                              #
# ======================================================================= #

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1 {{ color: #2255AA; }}
    h2 {{ color: #445577; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th {{ background: #2255AA; color: white; padding: 6px 10px; text-align: left; }}
    td {{ border: 1px solid #ccc; padding: 5px 10px; }}
    tr:nth-child(even) {{ background: #f4f7ff; }}
    .summary {{ background: #eef4ff; border: 1px solid #aabbdd; padding: 12px 20px;
               border-radius: 4px; margin-bottom: 24px; }}
    .summary table {{ margin: 0; }}
    .summary td {{ border: none; padding: 3px 10px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>

  <div class="summary">
    <h2>System Summary</h2>
    <table>
      <tr><td><b>Total Gain</b></td><td>{gain_db:.2f} dB</td></tr>
      <tr><td><b>Cascaded NF</b></td><td>{nf_db:.2f} dB</td></tr>
      <tr><td><b>System IIP3</b></td><td>{iip3}</td></tr>
      <tr><td><b>System OIP3</b></td><td>{oip3}</td></tr>
      <tr><td><b>Input P1dB</b></td><td>{p1db}</td></tr>
      <tr><td><b>Min Damage Level (at input)</b></td><td>{damage}</td></tr>
    </table>
  </div>

  <h2>Stage-by-Stage Analysis</h2>
  <table>
    <tr>
      <th>#</th><th>Block Type</th><th>Label</th>
      <th>Gain (dB)</th><th>NF (dB)</th>
      <th>Cum. Gain (dB)</th>
      <th>P1dB out (dBm)</th><th>OIP3 (dBm)</th>
      <th>Max Input (dBm)</th>
    </tr>
    {stage_rows}
  </table>
</body>
</html>
"""

_STAGE_ROW_TEMPLATE = (
    "<tr><td>{i}</td><td>{btype}</td><td>{label}</td>"
    "<td>{gain}</td><td>{nf}</td><td>{cum_gain}</td>"
    "<td>{p1db}</td><td>{oip3}</td><td>{max_in}</td></tr>"
)


def export_html_report(
    metrics: Dict,
    blocks: list,
    filepath: str = "report.html",
    title: str = "RF System Cascade Report",
) -> None:
    """
    Generate an HTML report summarizing cascade metrics.

    Parameters
    ----------
    metrics : dict
        Output of compute_cascade_metrics().
    blocks : list of RFBlock
    filepath : str
    title : str
    """
    stage_gains = metrics.get("stage_gains", [])
    stage_nfs = metrics.get("stage_nfs", [])
    cum_gains = metrics.get("cumulative_gains", [])

    stage_rows_html = []
    for i, block in enumerate(blocks):
        stage_rows_html.append(
            _STAGE_ROW_TEMPLATE.format(
                i=i + 1,
                btype=block.BLOCK_TYPE,
                label=block.label,
                gain=f"{stage_gains[i]:.2f}" if i < len(stage_gains) else "-",
                nf=f"{stage_nfs[i]:.2f}" if i < len(stage_nfs) else "-",
                cum_gain=f"{cum_gains[i]:.2f}" if i < len(cum_gains) else "-",
                p1db=(f"{block.p1db_dbm:.1f}" if block.p1db_dbm is not None else "N/A"),
                oip3=(f"{block.oip3_dbm:.1f}" if block.oip3_dbm is not None else "N/A"),
                max_in=(f"{block.max_input_power_dbm:.1f}"
                        if block.max_input_power_dbm is not None else "N/A"),
            )
        )

    def _fmt_dbm(val):
        return f"{val:.1f} dBm" if val is not None else "N/A"

    html = _REPORT_TEMPLATE.format(
        title=title,
        gain_db=metrics.get("gain_db", 0.0),
        nf_db=metrics.get("nf_db", 0.0),
        iip3=_fmt_dbm(metrics.get("iip3_dbm")),
        oip3=_fmt_dbm(metrics.get("oip3_dbm")),
        p1db=_fmt_dbm(metrics.get("p1db_in_dbm")),
        damage=_fmt_dbm(metrics.get("min_damage_dbm")),
        stage_rows="\n    ".join(stage_rows_html),
    )

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(html)
