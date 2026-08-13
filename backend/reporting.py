"""
Downloadable maintenance reports, in CSV and PDF.

Both are built in memory and streamed to the browser, and a copy is saved under
outputs/exports/ so there is a record of exactly what was handed over.

  CSV: one row per reading, for further work in Excel or pandas.
  PDF: a shift report with summary counts, the alerts that fired and their
       recommended actions, and a table of the most recent readings.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend import config, units

STATUS_COLOUR = {
    "Normal": colors.HexColor("#2e9e5b"),
    "Warning": colors.HexColor("#e0a800"),
    "Fault": colors.HexColor("#d9463f"),
    "Critical": colors.HexColor("#d9463f"),
    "Info": colors.HexColor("#6b7280"),
}

# Column names carry their unit, so nobody has to guess whether a temperature is
# kelvin or Celsius. The values are converted from the SI units stored
# internally. See backend/units.py.
CSV_COLUMNS = [
    "timestamp", "source", "status", "confidence",
    "air_temp_c", "process_temp_c", "temp_diff_c",
    "rot_speed_rpm", "torque_nm", "power_kw", "tool_wear_min", "strain_min_nm",
    "rul_min", "rul_binding",
    "product_type",
]


def _save_copy(filename: str, payload: bytes) -> None:
    """Keep a copy of each generated report under outputs/exports/."""
    try:
        config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.EXPORTS_DIR / filename).write_bytes(payload)
    except OSError:
        pass  # do not fail a download just because the archive copy failed


def timestamped_name(extension: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"machine-health-report-{stamp}.{extension}"


def build_csv(predictions: list[dict[str, Any]]) -> tuple[str, bytes]:
    """One row per reading, oldest first, which reads better chronologically."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in reversed(predictions):          # DB returns newest-first
        rul = row.get("rul_minutes")
        writer.writerow({
            "timestamp": row.get("timestamp"),
            "source": row.get("source"),
            "status": row.get("status"),
            "confidence": round(float(row.get("confidence", 0)), 4),
            "air_temp_c": round(units.kelvin_to_celsius(
                float(row.get("air_temp", 0))), 2),
            "process_temp_c": round(units.kelvin_to_celsius(
                float(row.get("process_temp", 0))), 2),
            # A temperature difference converts one to one, with no offset.
            "temp_diff_c": round(units.delta_kelvin_to_celsius(
                float(row.get("temp_diff", 0))), 2),
            "rot_speed_rpm": round(float(row.get("rot_speed", 0)), 1),
            "torque_nm": round(float(row.get("torque", 0)), 2),
            "power_kw": round(units.watts_to_kilowatts(
                float(row.get("power", 0))), 3),
            "tool_wear_min": round(float(row.get("tool_wear", 0)), 1),
            "strain_min_nm": round(float(row.get("strain", 0)), 1),
            # Predictions logged before RUL existed have NULL here, so leave the
            # cell empty rather than writing a misleading 0.
            "rul_min": "" if rul is None else round(float(rul), 1),
            "rul_binding": row.get("rul_binding"),
            "product_type": row.get("product_type"),
        })
    payload = buf.getvalue().encode()
    name = timestamped_name("csv")
    _save_copy(name, payload)
    return name, payload


def build_pdf(
    predictions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    *,
    username: str = "unknown",
    max_reading_rows: int = 40,
) -> tuple[str, bytes]:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Machine Health Report",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10,
                        spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.5,
                          leading=11, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=9.5)

    story: list = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    story.append(Paragraph("Machine Health &amp; Predictive Maintenance Report", h1))
    story.append(Paragraph(
        f"Generated {generated} &nbsp;|&nbsp; Requested by <b>{username}</b> "
        f"&nbsp;|&nbsp; {len(predictions)} readings, {len(alerts)} alerts", body))
    story.append(Spacer(1, 8))

    # ---------------- Summary ----------------
    counts = Counter(p.get("status", "Unknown") for p in predictions)
    sev_counts = Counter(a.get("severity", "Unknown") for a in alerts)
    total = max(len(predictions), 1)

    story.append(Paragraph("1. Summary", h2))
    summary_rows = [["Status", "Readings", "Share"]]
    for status in ("Normal", "Warning", "Fault"):
        n = counts.get(status, 0)
        summary_rows.append([status, str(n), f"{n / total * 100:.1f}%"])
    summary_rows.append(["", "", ""])
    summary_rows.append(["Critical alerts", str(sev_counts.get("Critical", 0)), ""])
    summary_rows.append(["Warning alerts", str(sev_counts.get("Warning", 0)), ""])

    tbl = Table(summary_rows, colWidths=[60 * mm, 30 * mm, 30 * mm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]
    for i, status in enumerate(("Normal", "Warning", "Fault"), start=1):
        style.append(("TEXTCOLOR", (0, i), (0, i), STATUS_COLOUR[status]))
        style.append(("FONTNAME", (0, i), (0, i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)

    # ---------------- Alerts + recommended actions ----------------
    story.append(Paragraph("2. Alerts and recommended actions", h2))
    if not alerts:
        story.append(Paragraph("No alerts were raised in this period.", body))
    else:
        rows = [["Time (UTC)", "Severity", "Condition", "Recommended action"]]
        for a in alerts[:30]:
            rows.append([
                Paragraph(str(a.get("timestamp", ""))[:19].replace("T", " "), small),
                Paragraph(f"<b>{a.get('severity','')}</b>", small),
                Paragraph(str(a.get("title", "")), small),
                Paragraph(str(a.get("recommended_action", "")), small),
            ])
        atbl = Table(rows, colWidths=[26 * mm, 17 * mm, 42 * mm, 93 * mm],
                     repeatRows=1)
        astyle = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for i, a in enumerate(alerts[:30], start=1):
            astyle.append(
                ("TEXTCOLOR", (1, i), (1, i),
                 STATUS_COLOUR.get(a.get("severity", ""), colors.black))
            )
        atbl.setStyle(TableStyle(astyle))
        story.append(atbl)

    # ---------------- Readings ----------------
    story.append(PageBreak())
    story.append(Paragraph(
        f"3. Recent sensor readings (most recent {max_reading_rows})", h2))
    story.append(Paragraph(
        "ΔT = process temp - air temp (a difference of 10 K is 10 °C). "
        "Power = torque × angular velocity. Strain = tool wear × torque. "
        "RUL = remaining cutting minutes before the first binding limit "
        "(tool wear, or overstrain at the current torque). Values are stored "
        "internally in SI units (K, W) and converted for display.",
        small))
    story.append(Spacer(1, 4))

    head = ["Time (UTC)", "Status", "Conf.", "Air °C", "Proc °C", "ΔT °C",
            "rpm", "N·m", "kW", "Wear min", "RUL min"]
    rrows = [head]
    for p in predictions[:max_reading_rows]:
        rul = p.get("rul_minutes")
        rrows.append([
            str(p.get("timestamp", ""))[11:19],
            str(p.get("status", "")),
            f"{float(p.get('confidence', 0)) * 100:.0f}%",
            f"{units.kelvin_to_celsius(float(p.get('air_temp', 0))):.1f}",
            f"{units.kelvin_to_celsius(float(p.get('process_temp', 0))):.1f}",
            f"{units.delta_kelvin_to_celsius(float(p.get('temp_diff', 0))):.1f}",
            f"{float(p.get('rot_speed', 0)):.0f}",
            f"{float(p.get('torque', 0)):.1f}",
            f"{units.watts_to_kilowatts(float(p.get('power', 0))):.2f}",
            f"{float(p.get('tool_wear', 0)):.0f}",
            "—" if rul is None else f"{float(rul):.0f}",
        ])

    rtbl = Table(rrows, repeatRows=1, colWidths=[19 * mm, 16 * mm, 12 * mm, 15 * mm,
                                                 16 * mm, 12 * mm, 16 * mm, 13 * mm,
                                                 18 * mm, 17 * mm, 15 * mm])
    rstyle = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ]
    for i, p in enumerate(predictions[:max_reading_rows], start=1):
        rstyle.append(("TEXTCOLOR", (1, i), (1, i),
                       STATUS_COLOUR.get(p.get("status", ""), colors.black)))
        if i % 2 == 0:
            rstyle.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9fafb")))
    rtbl.setStyle(TableStyle(rstyle))
    story.append(rtbl)

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Predictions come from a Random Forest classifier trained on the AI4I 2020 "
        "predictive-maintenance dataset, combined with deterministic physical "
        "threshold rules. Threshold rules can escalate the model's verdict but "
        "never suppress it. This report is generated from the system's audit log.",
        small))

    doc.build(story)
    payload = buf.getvalue()
    name = timestamped_name("pdf")
    _save_copy(name, payload)
    return name, payload


def normalise_alert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Database rows store triggered_rules as a JSON string. Decode for the API."""
    out = []
    for r in rows:
        r = dict(r)
        raw = r.get("triggered_rules")
        if isinstance(raw, str):
            try:
                r["triggered_rules"] = json.loads(raw)
            except json.JSONDecodeError:
                r["triggered_rules"] = []
        out.append(r)
    return out
