#!/usr/bin/env python3
"""
Usage:
    python3 generate_wound_report.py ~/wound_reports/2026-04-26_10-00-00
Generates report.pdf inside the session directory.
Requires: pip install reportlab
"""
import sys
import os
import json

try:
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    MARGIN    = 1.5 * cm
    CONTENT_W = A4[0] - 2 * MARGIN
    CROP_W = 5.0 * cm
    CROP_H = 5.0 * cm
    SKEL_W = 12.0 * cm
    SKEL_H = 7.5 * cm
    RED = colors.HexColor('#c0392b')
    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False


def _img(path, w, h, styles):
    if os.path.exists(path):
        return RLImage(path, width=w, height=h)
    return Paragraph('<i>image not found</i>', styles['Normal'])


def _info_table(w_data):
    pose = w_data.get('pose_base_link', {})
    pose_str = (f"x={pose['x']:.3f}  y={pose['y']:.3f}  z={pose['z']:.3f} m"
                if pose else '—')
    loc = w_data['body_location'].replace('_', ' ').title()

    left_rows = [
        ['Location',   loc],
        ['Depth',      f"{w_data['depth_m']:.2f} m"],
        ['Size',       f"{w_data['width_mm']:.1f} × {w_data['height_mm']:.1f} mm"],
        ['Area',       f"~{w_data['area_cm2']:.1f} cm²"],
    ]
    right_rows = [
        ['Class',      f"{w_data['label']} ({w_data['confidence']*100:.0f}%)"],
        ['Redness',    f"{w_data['redness_index']:.2f}"],
        ['Robot pose', pose_str],
    ]

    style = TableStyle([
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('FONTNAME',      (0, 0), (0, -1),  'Helvetica-Bold'),
        ('TEXTCOLOR',     (0, 0), (0, -1),  colors.HexColor('#555555')),
        ('ROWBACKGROUNDS',(0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID',          (0, 0), (-1, -1), 0.3, colors.HexColor('#dddddd')),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
    ])

    half = CONTENT_W / 2
    kw   = [2.8 * cm, half - 2.8 * cm - 4]

    def _mini(rows):
        t = Table(rows, colWidths=kw)
        t.setStyle(style)
        return t

    combined = Table([[_mini(left_rows), _mini(right_rows)]], colWidths=[half, half])
    combined.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return combined


def _wound_elements(w, session_dir, styles):
    loc      = w['body_location'].replace('_', ' ').title()
    crop_img = _img(os.path.join(session_dir, w['images']['crop']),     CROP_W, CROP_H, styles)
    skel_img = _img(os.path.join(session_dir, w['images']['skeleton']), SKEL_W, SKEL_H, styles)

    h3 = ParagraphStyle('h3', parent=styles['Normal'],
                         fontSize=11, fontName='Helvetica-Bold',
                         textColor=RED, spaceAfter=5)

    img_table = Table(
        [[crop_img, skel_img]],
        colWidths=[CROP_W + 0.5 * cm, CONTENT_W - CROP_W - 0.5 * cm]
    )
    img_table.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    return [
        Paragraph(f'Wound #{w["id"]} — {loc}', h3),
        img_table,
        _info_table(w),
        Spacer(1, 0.5 * cm),
        HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')),
        Spacer(1, 0.5 * cm),
    ]


def _generate(session_dir):
    data_path = os.path.join(session_dir, 'session_data.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')

    with open(data_path) as f:
        data = json.load(f)

    session_id = data.get('session_id', 'unknown')
    wounds     = data.get('wounds', [])
    updated    = data.get('last_updated', '—')
    out_path   = os.path.join(session_dir, 'report.pdf')

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             leftMargin=MARGIN, rightMargin=MARGIN,
                             topMargin=MARGIN,  bottomMargin=MARGIN)
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', parent=styles['Normal'],
                              fontSize=18, fontName='Helvetica-Bold',
                              textColor=RED, spaceAfter=4)
    sub_s   = ParagraphStyle('S', parent=styles['Normal'],
                              fontSize=9, textColor=colors.grey, spaceAfter=10)

    story = [
        Paragraph('Wound Assessment Report', title_s),
        Paragraph(f'Session: {session_id} &nbsp;|&nbsp; '
                  f'{len(wounds)} wound(s) &nbsp;|&nbsp; Updated: {updated}', sub_s),
        HRFlowable(width='100%', thickness=1.2, color=RED),
        Spacer(1, 0.6 * cm),
    ]

    if not wounds:
        story.append(Paragraph('No wounds recorded in this session.', styles['Normal']))
    else:
        for w in wounds:
            story.extend(_wound_elements(w, session_dir, styles))

    doc.build(story)
    print(f'Report saved: {out_path}')


def generate(session_dir):
    if not _REPORTLAB_OK:
        raise RuntimeError('reportlab not installed — run: pip install reportlab')
    _generate(session_dir)


if __name__ == '__main__':
    if not _REPORTLAB_OK:
        print('reportlab not found — install it:  pip install reportlab')
        sys.exit(1)
    if len(sys.argv) < 2:
        print('Usage: generate_wound_report.py <session_dir>')
        sys.exit(1)
    generate(os.path.expanduser(sys.argv[1]))
