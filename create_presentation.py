"""
create_presentation.py
Generates outputs/pitch_deck.pptx for SanDisk Cerebrum 2026 Die Yield Prediction.
"""

from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
SHAP_DIR = OUTPUT_DIR / "shap"
WAFER_MAPS_DIR = OUTPUT_DIR / "wafer_maps"

# Color Palette (Dark Executive Tech Theme)
COLOR_BG = RGBColor(11, 17, 32)          # #0B1120 Deep Navy
COLOR_CARD = RGBColor(30, 41, 59)        # #1E293B Slate 800
COLOR_CARD_BORDER = RGBColor(51, 65, 85) # #334155 Slate 700
COLOR_CYAN = RGBColor(56, 189, 248)      # #38BDF8 Sky Blue
COLOR_AMBER = RGBColor(245, 158, 11)     # #F59E0B Amber
COLOR_GREEN = RGBColor(34, 197, 94)      # #22C55E Emerald Green
COLOR_RED = RGBColor(239, 68, 68)        # #EF4444 Coral Red
COLOR_WHITE = RGBColor(248, 250, 252)    # #F8FAFC White
COLOR_MUTED = RGBColor(148, 163, 184)    # #94A3B8 Muted Gray
COLOR_BODY = RGBColor(203, 213, 225)     # #CBD5E1 Light Slate
FONT_NAME = "Calibri"


def set_slide_background(slide):
    """Draw a dark background rectangle over the entire slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.color.rgb = COLOR_BG
    return bg


def add_header(slide, category: str, headline: str, subtitle: str):
    """Add standard structured header to slide."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.2))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Category tracker
    p0 = tf.paragraphs[0]
    p0.space_after = Pt(2)
    r0 = p0.add_run()
    r0.text = category.upper()
    r0.font.name = FONT_NAME
    r0.font.size = Pt(10)
    r0.font.bold = True
    r0.font.color.rgb = COLOR_CYAN

    # Main Headline
    p1 = tf.add_paragraph()
    p1.space_after = Pt(2)
    r1 = p1.add_run()
    r1.text = headline
    r1.font.name = FONT_NAME
    r1.font.size = Pt(22)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_WHITE

    # Subtitle
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = subtitle
    r2.font.name = FONT_NAME
    r2.font.size = Pt(12)
    r2.font.color.rgb = COLOR_MUTED


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Main Title Box
    title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.15), Inches(11.333), Inches(2.3))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.space_after = Pt(8)
    r_badge = p_badge.add_run()
    r_badge.text = "SANDISK CEREBRUM 2026 | FINAL COMPETITION PITCH"
    r_badge.font.name = FONT_NAME
    r_badge.font.size = Pt(11)
    r_badge.font.bold = True
    r_badge.font.color.rgb = COLOR_CYAN

    p_main = tf1.add_paragraph()
    p_main.space_after = Pt(8)
    r_main = p_main.add_run()
    r_main.text = "Die Yield Prediction"
    r_main.font.name = FONT_NAME
    r_main.font.size = Pt(40)
    r_main.font.bold = True
    r_main.font.color.rgb = COLOR_WHITE

    p_sub = tf1.add_paragraph()
    p_sub.space_after = Pt(14)
    r_sub = p_sub.add_run()
    r_sub.text = "Pre-Screening Post-Burn-In Silicon Failures via Spatial Risk Fields & Sub-Die Diagnostics"
    r_sub.font.name = FONT_NAME
    r_sub.font.size = Pt(17)
    r_sub.font.color.rgb = COLOR_MUTED

    p_team = tf1.add_paragraph()
    r_team_label = p_team.add_run()
    r_team_label.text = "Team: "
    r_team_label.font.name = FONT_NAME
    r_team_label.font.size = Pt(13)
    r_team_label.font.bold = True
    r_team_label.font.color.rgb = COLOR_AMBER

    r_team = p_team.add_run()
    r_team.text = "Cookies of the Dark Web  "
    r_team.font.name = FONT_NAME
    r_team.font.size = Pt(13)
    r_team.font.bold = True
    r_team.font.color.rgb = COLOR_WHITE

    r_members = p_team.add_run()
    r_members.text = "(Shrey Sharma, Aditya Gupta, Prisha Kushwaha, Darsh Ramoliya)"
    r_members.font.name = FONT_NAME
    r_members.font.size = Pt(13)
    r_members.font.color.rgb = COLOR_BODY

    # 4 Key Pillars Cards across bottom
    card_w = Inches(2.68)
    card_h = Inches(2.35)
    card_y = Inches(4.35)
    pillars = [
        ("INNOVATION & ARCHITECTURE", "Multi-Resolution Ranking", "Continuous 2D spatial hazard surfaces (sigma=1.5) combined with 2,000-reading sub-die block telemetry to detect micro-structural anomalies."),
        ("TECHNICAL RIGOR", "Rigorous Statistical Auditing", "Multi-seed paired ablation and wafer-cluster bootstrap prove ranking gain (+0.0337 PR-AUC, p < 0.001) while F1 is within noise."),
        ("EXPLAINABILITY & UX", "Hierarchical Interpretability", "Wafer maps -> Die inspection -> TreeSHAP attributions -> Sub-die block profiles (separating spatial context from actionable signal)."),
        ("OPERATIONAL VALUE", "Screening Triage & Reliability", "+64 failures captured at 10% budget; validation-only Platt scaling cuts calibration error by over 70% with zero test leakage."),
    ]

    for i, (tag, head, desc) in enumerate(pillars):
        cx = Inches(1.0 + i * 2.88)
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, card_y, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = COLOR_CARD_BORDER

        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = Inches(0.18)

        cp0 = ctf.paragraphs[0]
        cp0.space_after = Pt(4)
        cr0 = cp0.add_run()
        cr0.text = tag
        cr0.font.name = FONT_NAME
        cr0.font.size = Pt(8.5)
        cr0.font.bold = True
        cr0.font.color.rgb = COLOR_CYAN

        cp1 = ctf.add_paragraph()
        cp1.space_after = Pt(6)
        cr1 = cp1.add_run()
        cr1.text = head
        cr1.font.name = FONT_NAME
        cr1.font.size = Pt(12)
        cr1.font.bold = True
        cr1.font.color.rgb = COLOR_WHITE

        cp2 = ctf.add_paragraph()
        cr2 = cp2.add_run()
        cr2.text = desc
        cr2.font.name = FONT_NAME
        cr2.font.size = Pt(9.5)
        cr2.font.color.rgb = COLOR_BODY

    # =========================================================================
    # SLIDE 2: Problem + Approach Combined
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(
        s2,
        category="Problem & Approach",
        headline="The Die Yield Task: Macro Spatial Context vs. Micro Sub-Die Signals",
        subtitle="Predicting the 4.23% post-burn-in defects early without over-scrapping healthy silicon"
    )

    col_w = Inches(3.64)
    col_h = Inches(5.1)
    col_y = Inches(1.8)

    # Box 1: Problem & Population
    b1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), col_y, col_w, col_h)
    b1.fill.solid()
    b1.fill.fore_color.rgb = COLOR_CARD
    b1.line.color.rgb = COLOR_CARD_BORDER
    tf_b1 = b1.text_frame
    tf_b1.word_wrap = True
    tf_b1.margin_left = tf_b1.margin_top = tf_b1.margin_right = tf_b1.margin_bottom = Inches(0.25)

    p = tf_b1.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "1. THE FACTORY CHALLENGE"
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = COLOR_AMBER

    p = tf_b1.add_paragraph()
    p.space_after = Pt(12)
    r = p.add_run()
    r.text = "Post-Burn-In Defect Prediction"
    r.font.name = FONT_NAME
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    bullets_b1 = [
        ("Silicon Wafer Scale: ", "Silicon chips (dies) are tested before packaging. Post-burn-in testing is slow and factory-constrained; failures are spatially clustered."),
        ("Pre-Test Exclusions: ", "Known defective chips (old_label=1; 6,753 dies on test set) are already dead and excluded per competition evaluation rule."),
        ("The True Target (Eligible Dies): ", "Exactly 32,598 eligible dies (old_label=0) on holdout test set with a true post-test defect rate of 4.23% (1,380 fails)."),
        ("Core Scientific Question: ", "Does adding sub-die block telemetry improve risk ranking and screening efficiency beyond macro spatial context?"),
    ]
    for b_title, b_desc in bullets_b1:
        p = tf_b1.add_paragraph()
        p.space_after = Pt(8)
        r_bt = p.add_run()
        r_bt.text = b_title
        r_bt.font.name = FONT_NAME
        r_bt.font.bold = True
        r_bt.font.size = Pt(10)
        r_bt.font.color.rgb = COLOR_CYAN
        r_bd = p.add_run()
        r_bd.text = b_desc
        r_bd.font.name = FONT_NAME
        r_bd.font.size = Pt(9.5)
        r_bd.font.color.rgb = COLOR_BODY

    # Box 2: Model A (Die + Spatial)
    b2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.84), col_y, col_w, col_h)
    b2.fill.solid()
    b2.fill.fore_color.rgb = COLOR_CARD
    b2.line.color.rgb = COLOR_CARD_BORDER
    tf_b2 = b2.text_frame
    tf_b2.word_wrap = True
    tf_b2.margin_left = tf_b2.margin_top = tf_b2.margin_right = tf_b2.margin_bottom = Inches(0.25)

    p = tf_b2.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "2. MODEL A (MACRO + SPATIAL)"
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN

    p = tf_b2.add_paragraph()
    p.space_after = Pt(12)
    r = p.add_run()
    r.text = "511 Leakage-Free Features"
    r.font.name = FONT_NAME
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    bullets_b2 = [
        ("500 Parametric Sensors: ", "Full-die electrical measurements from automated test equipment probe cards."),
        ("10 Spatial Features: ", "Leakage-free wafer geography: distance to nearest pre-test fail (sp_dist_to_fail), edge proximity, and multi-radius fail counts."),
        ("1 Die Anomaly Score: ", "Isolation Forest fitted strictly on healthy chips to capture multivariate out-of-distribution drift."),
        ("Algorithm & Threshold: ", "LightGBM Classifier (num_leaves=63, scale_pos_weight=12.0), independently tuned threshold = 0.5574."),
    ]
    for b_title, b_desc in bullets_b2:
        p = tf_b2.add_paragraph()
        p.space_after = Pt(8)
        r_bt = p.add_run()
        r_bt.text = b_title
        r_bt.font.name = FONT_NAME
        r_bt.font.bold = True
        r_bt.font.size = Pt(10)
        r_bt.font.color.rgb = COLOR_CYAN
        r_bd = p.add_run()
        r_bd.text = b_desc
        r_bd.font.name = FONT_NAME
        r_bd.font.size = Pt(9.5)
        r_bd.font.color.rgb = COLOR_BODY

    # Box 3: Model B (Die + Spatial + Block)
    b3 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.88), col_y, col_w, col_h)
    b3.fill.solid()
    b3.fill.fore_color.rgb = COLOR_CARD
    b3.line.color.rgb = COLOR_CARD_BORDER
    tf_b3 = b3.text_frame
    tf_b3.word_wrap = True
    tf_b3.margin_left = tf_b3.margin_top = tf_b3.margin_right = tf_b3.margin_bottom = Inches(0.25)

    p = tf_b3.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "3. MODEL B (DEEP SUB-DIE TELEMETRY)"
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = COLOR_GREEN

    p = tf_b3.add_paragraph()
    p.space_after = Pt(12)
    r = p.add_run()
    r.text = "531 Features (Model A + Blocks)"
    r.font.name = FONT_NAME
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    bullets_b3 = [
        ("Full Model A Inheritance: ", "All 511 macro electrical, spatial neighborhood, and die anomaly features."),
        ("8 Block Summary Stats: ", "Aggregations over 2,000 sub-die block readings per chip: blk_mean, blk_std, blk_q25, blk_q75, etc."),
        ("1 Block Anomaly Score: ", "Robust MAD-based anomaly detector measuring localized sub-die variance shifts (|x - median| > 2*MAD)."),
        ("11 Sub-Die Telemetry Features: ", "Extended block variance percentiles; LightGBM threshold = 0.5180 with permanent index alignment."),
    ]
    for b_title, b_desc in bullets_b3:
        p = tf_b3.add_paragraph()
        p.space_after = Pt(8)
        r_bt = p.add_run()
        r_bt.text = b_title
        r_bt.font.name = FONT_NAME
        r_bt.font.bold = True
        r_bt.font.size = Pt(10)
        r_bt.font.color.rgb = COLOR_GREEN
        r_bd = p.add_run()
        r_bd.text = b_desc
        r_bd.font.name = FONT_NAME
        r_bd.font.size = Pt(9.5)
        r_bd.font.color.rgb = COLOR_BODY

    # =========================================================================
    # SLIDE 3: Benchmark Results
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(
        s3,
        category="Benchmark Results & Rigorous Auditing",
        headline="Production Results: PR-AUC Gain is Statistically Real; F1 Gain is Within Noise",
        subtitle="5-seed paired ablation and 1,000-iteration bootstrap prove the continuous ranking power of block signals"
    )

    # Comparison Table
    table_shape = s3.shapes.add_table(6, 5, Inches(0.8), Inches(1.8), Inches(8.0), Inches(3.2))
    table = table_shape.table

    table_data = [
        ["Metric (Eligible Dies)", "Model A (511)", "Model B (531)", "Delta (B - A)", "Statistical Test (5-Seed / Bootstrap)"],
        ["PR-AUC (Secondary)", "0.5025", "0.5362", "+0.0337 (+6.7%)", "p = 1.33e-5 (Statistically Real, CI [+0.023, +0.043] > 0)"],
        ["Fail F1 Score (Primary)", "0.5207", "0.5222", "+0.0015 (+0.3%)", "p = 0.1081 (Within Noise, CI [-0.008, +0.012] crosses 0)"],
        ["Fail Recall", "35.58% (491/1380)", "37.10% (512/1380)", "+1.52% (+21 dies)", "p = 0.0090 (Statistically Significant)"],
        ["Fail Precision", "97.04% (491/506)", "88.12% (512/581)", "-8.92% (69 FP)", "Controlled Fab Scrap (<0.22% False Alarm Rate)"],
        ["Top-10% Triage Capture", "55.7% (768 fails)", "60.4% (834 fails)", "+4.8% (+66 fails)", "+66 Defective Dies Caught (Same Budget)"],
    ]

    col_widths = [Inches(2.0), Inches(1.3), Inches(1.4), Inches(1.4), Inches(1.9)]
    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
            elif i in [1, 2]:
                cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
            else:
                cell.fill.fore_color.rgb = RGBColor(24, 33, 47)

            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.05)

            cp = cell.text_frame.paragraphs[0]
            cp.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            cr = cp.add_run()
            cr.text = val
            cr.font.name = FONT_NAME
            cr.font.size = Pt(8.5 if i > 0 else 9.0)
            if i == 0:
                cr.font.bold = True
                cr.font.color.rgb = COLOR_CYAN
            elif j == 0:
                cr.font.bold = True
                cr.font.color.rgb = COLOR_WHITE
            elif j == 3:
                cr.font.bold = True
                if "+" in val:
                    cr.font.color.rgb = COLOR_GREEN
                elif "-" in val:
                    cr.font.color.rgb = COLOR_AMBER
                else:
                    cr.font.color.rgb = COLOR_BODY
            elif j == 4 and "Real" in val:
                cr.font.bold = True
                cr.font.color.rgb = COLOR_GREEN
            elif j == 4 and "Noise" in val:
                cr.font.bold = True
                cr.font.color.rgb = COLOR_AMBER
            else:
                cr.font.color.rgb = COLOR_BODY

    # Callout card underneath table
    callout = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.2), Inches(8.0), Inches(1.7))
    callout.fill.solid()
    callout.fill.fore_color.rgb = COLOR_CARD
    callout.line.color.rgb = COLOR_AMBER
    tf_co = callout.text_frame
    tf_co.word_wrap = True
    tf_co.margin_left = tf_co.margin_top = tf_co.margin_right = tf_co.margin_bottom = Inches(0.18)

    p = tf_co.paragraphs[0]
    p.space_after = Pt(3)
    r = p.add_run()
    r.text = "KEY SCIENTIFIC FINDING: MARGINAL DEFECT DISTRIBUTION OVERLAP"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_AMBER

    p = tf_co.add_paragraph()
    r = p.add_run()
    r.text = (
        "- Model B is NOT a binary F1 boundary improver. It is sold as improving continuous probability ranking and screening efficiency.\n"
        "- Sub-die block telemetry provides continuous gradient signal that significantly improves ranking (PR-AUC +0.0337, bootstrap CI > 0).\n"
        "- Fail F1 remains essentially unchanged (+0.0015, 95% wafer-cluster bootstrap CI [-0.008, +0.012] crosses zero).\n"
        "- Empirical Finding: A large fraction of failures (65% in this semi-synthetic design) are marginal and overlap healthy electrical distributions, making binary classification difficult; observed F1 remains around 0.52."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_BODY

    # Right Side: Embed Real Image from outputs/shap/shap_summary_model_b.png
    shap_img_path = SHAP_DIR / "shap_summary_model_b.png"
    if shap_img_path.exists():
        s3.shapes.add_picture(str(shap_img_path), Inches(9.1), Inches(1.8), width=Inches(3.43), height=Inches(5.1))

    # =========================================================================
    # SLIDE 4: Interpretability & Wafer Risk Fields
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(
        s4,
        category="Interpretability & Visual Design",
        headline="Hierarchical Interpretability: Wafer Risk Surfaces -> Sub-Die Evidence",
        subtitle="Gaussian smoothing (sigma=1.5) delineates hazard contours; TreeSHAP isolates actionable signal from spatial context"
    )

    # Embed Real Image 1 from outputs/plots/risk_field_wafer_W_F_0014.png
    rf_img_path = PLOTS_DIR / "risk_field_wafer_W_F_0014.png"
    if rf_img_path.exists():
        s4.shapes.add_picture(str(rf_img_path), Inches(0.8), Inches(1.8), width=Inches(11.733), height=Inches(2.35))

    # Embed Real Image 2 from outputs/wafer_maps/wafer_W_F_0014_model_b.png
    wm_img_path = WAFER_MAPS_DIR / "wafer_W_F_0014_model_b.png"
    if wm_img_path.exists():
        s4.shapes.add_picture(str(wm_img_path), Inches(0.8), Inches(4.25), width=Inches(11.733), height=Inches(1.5))

    # Teaser banner at bottom
    teaser = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.9), Inches(11.733), Inches(1.15))
    teaser.fill.solid()
    teaser.fill.fore_color.rgb = COLOR_CARD
    teaser.line.color.rgb = COLOR_CYAN
    tf_ts = teaser.text_frame
    tf_ts.word_wrap = True
    tf_ts.margin_left = tf_ts.margin_top = tf_ts.margin_right = tf_ts.margin_bottom = Inches(0.15)

    p = tf_ts.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "WHAT THE LIVE INTERACTIVE DASHBOARD REVEALS (BEYOND STATIC SLIDES):"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN

    p = tf_ts.add_paragraph()
    r = p.add_run()
    r.text = (
        "- Multi-Resolution Die Inspection: Preset W_N_0066 die (2,13) where Model A raw score is 12.9% PASS (calibrated: 4.5%), but Model B raw score is 62.9% FAIL (calibrated: 45.2%) due to sub-die block variance.\n"
        "- Context vs Actionable Signal: 'Spatial features provide context, not an intervention.' Actionable signal lies in die parametric & sub-die block readings.\n"
        "- 4 HDBSCAN Failure Signatures: Memory Array Reading Shift, Extreme Block Anomaly, Defect Neighborhood Proximity, Edge Drift.\n"
        "- 2,000-Point Sub-Die Block Strips: In-depth inspection of sub-die memory array voltage profiles using robust MAD anomaly highlighting."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(9.0)
    r.font.color.rgb = COLOR_BODY

    # =========================================================================
    # SLIDE 5: Operational Triage & Risk Calibration
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(
        s5,
        category="Operational Triage, Calibration & Fab Economics",
        headline="Operational Triage & Reliability: Screening Efficiency + Platt Calibration",
        subtitle="Optimized for volume semiconductor manufacturing: bounded screening triage, calibrated risk, zero blind scrap"
    )

    card_w5 = Inches(5.72)
    card_h5 = Inches(2.4)

    # Card 1: Operational Screening Triage
    c1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), card_w5, card_h5)
    c1.fill.solid()
    c1.fill.fore_color.rgb = COLOR_CARD
    c1.line.color.rgb = COLOR_CARD_BORDER
    tf = c1.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.22)

    p = tf.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "OPERATIONAL SCREENING TRIAGE"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN

    p = tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run()
    r.text = "+66 Failures Caught at 10% Budget"
    r.font.name = FONT_NAME
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = (
        "- At the same 10% inspection budget (3,260 dies inspected): Model A catches 55.7% (768 fails) vs. Model B catches 60.4% (834 fails).\n"
        "- Captures 66 more failures without increasing the inspection budget, maximizing factory throughput.\n"
        "- Top 2% triage: 38.1% (A) vs 38.8% (B); Top 5% triage: 46.2% (A) vs 49.7% (B)."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BODY

    # Card 2: Validation-Only Platt Probability Calibration
    c2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.81), Inches(1.8), card_w5, card_h5)
    c2.fill.solid()
    c2.fill.fore_color.rgb = COLOR_CARD
    c2.line.color.rgb = COLOR_CARD_BORDER
    tf = c2.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.22)

    p = tf.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "PROBABILITY CALIBRATION (PLATT SCALING)"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_GREEN

    p = tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run()
    r.text = "> 70% Calibration Error Reduction"
    r.font.name = FONT_NAME
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = (
        "- Fitted strictly on 5 validation folds (zero test data leakage); evaluated on 32,598 held-out test dies.\n"
        "- Model A ECE: 0.0336 -> 0.0084 (-74.9%); Model B ECE: 0.0293 -> 0.0085 (-71.0%).\n"
        "- Monotonic mapping perfectly preserves ranking & PR-AUC while providing reliable true defect probabilities."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BODY

    # Card 3: Controlled Fab Scrap & Precision
    c3 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.5), card_w5, card_h5)
    c3.fill.solid()
    c3.fill.fore_color.rgb = COLOR_CARD
    c3.line.color.rgb = COLOR_CARD_BORDER
    tf = c3.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.22)

    p = tf.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "SCRAP-PRESERVING HIGH PRECISION"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_AMBER

    p = tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run()
    r.text = "88.12% Precision / < 0.22% False Alarms"
    r.font.name = FONT_NAME
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = (
        "- Only 69 false alarms out of 31,218 healthy dies across 40 test wafers (99.78% pass specificity; Model A: 15 FP).\n"
        "- Deliberate engineering setting: scale_pos_weight = 12.0 prevents violent gradient instability.\n"
        "- Never over-scraps expensive, functional silicon dies, preserving crucial semiconductor fab margins."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BODY

    # Card 4: In-Line Deployability
    c4 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.81), Inches(4.5), card_w5, card_h5)
    c4.fill.solid()
    c4.fill.fore_color.rgb = COLOR_CARD
    c4.line.color.rgb = COLOR_CARD_BORDER
    tf = c4.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Inches(0.22)

    p = tf.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "IN-LINE FACTORY DEPLOYABILITY"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN

    p = tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run()
    r.text = "< 0.8s Inference per 40-Wafer Lot"
    r.font.name = FONT_NAME
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = (
        "- All 39,351 dies scored in under 0.8 seconds on a single CPU thread without GPU infrastructure.\n"
        "- Fits directly into automated test equipment (ATE) probe card software without adding cycle time.\n"
        "- Modular src/ library with automated index alignment assertions prevents silent data slicing errors."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_BODY

    # =========================================================================
    # SLIDE 6: Trade-offs/Limitations + Large Clickable Links
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(
        s6,
        category="Limitations & Interactive Links",
        headline="Technical Limitations, Real-Fab Horizons & Interactive Demo",
        subtitle="Transparent boundary conditions, reproducible open-source code, and live cloud deployment"
    )

    lim_box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.9))
    lim_box.fill.solid()
    lim_box.fill.fore_color.rgb = COLOR_CARD
    lim_box.line.color.rgb = COLOR_CARD_BORDER
    tf_lim = lim_box.text_frame
    tf_lim.word_wrap = True
    tf_lim.margin_left = tf_lim.margin_top = tf_lim.margin_right = tf_lim.margin_bottom = Inches(0.25)

    p = tf_lim.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "HONEST TECHNICAL AUDIT & BOUNDARIES"
    r.font.name = FONT_NAME
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = COLOR_AMBER

    p = tf_lim.add_paragraph()
    p.space_after = Pt(12)
    r = p.add_run()
    r.text = "Engineering Limitations & Next Steps"
    r.font.name = FONT_NAME
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    limits = [
        ("1. Semi-Synthetic Benchmark: ", "Built on real WM-811K wafer geometries with synthetic sensor and block telemetry. Validating on real fab telemetry is the natural next milestone; no claim of immediate production readiness."),
        ("2. Marginal Defect Distribution Overlap: ", "A large fraction of failures (65% in this semi-synthetic design) are marginal and overlap healthy electrical distributions, making binary classification difficult; observed F1 remains around 0.52 without physical 3D block coordinates."),
        ("3. Spatial Context is Non-Interventionist: ", "Spatial features provide context, not an intervention. Spatial clustering guides inspection prioritization, but physical fab actions require process parameter changes."),
        ("4. Inert 1D Block Telemetry: ", "Block readings were provided as 1D scalar streams (0..1999); preserving exact 3D physical (x, y, z) memory array coordinates would enable 3D CNNs to capture spatial word-line leakage."),
    ]
    for l_title, l_desc in limits:
        p = tf_lim.add_paragraph()
        p.space_after = Pt(10)
        r_lt = p.add_run()
        r_lt.text = l_title
        r_lt.font.name = FONT_NAME
        r_lt.font.bold = True
        r_lt.font.size = Pt(10)
        r_lt.font.color.rgb = COLOR_CYAN
        r_ld = p.add_run()
        r_ld.text = l_desc
        r_ld.font.name = FONT_NAME
        r_ld.font.size = Pt(9.5)
        r_ld.font.color.rgb = COLOR_BODY

    # Right Column: Two Large Clickable Cards + Closing
    link_card1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(1.8), Inches(5.833), Inches(2.2))
    link_card1.fill.solid()
    link_card1.fill.fore_color.rgb = COLOR_CARD
    link_card1.line.color.rgb = COLOR_CYAN
    tf_lc1 = link_card1.text_frame
    tf_lc1.word_wrap = True
    tf_lc1.margin_left = tf_lc1.margin_top = tf_lc1.margin_right = tf_lc1.margin_bottom = Inches(0.2)

    p = tf_lc1.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "LIVE CLOUD DEPLOYMENT"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN

    p = tf_lc1.add_paragraph()
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "Streamlit Interactive Web Application"
    r.font.name = FONT_NAME
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf_lc1.add_paragraph()
    p.space_after = Pt(8)
    r = p.add_run()
    r.text = "Explore 40 test wafers, continuous risk fields, per-die SHAP breakdowns, and sub-die block profiles in real time."
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.color.rgb = COLOR_BODY

    p = tf_lc1.add_paragraph()
    r = p.add_run()
    r.text = "https://die-yield-prediction.streamlit.app/"
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = COLOR_CYAN
    r.hyperlink.address = "https://die-yield-prediction.streamlit.app/"

    link_card2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(4.2), Inches(5.833), Inches(1.5))
    link_card2.fill.solid()
    link_card2.fill.fore_color.rgb = COLOR_CARD
    link_card2.line.color.rgb = COLOR_GREEN
    tf_lc2 = link_card2.text_frame
    tf_lc2.word_wrap = True
    tf_lc2.margin_left = tf_lc2.margin_top = tf_lc2.margin_right = tf_lc2.margin_bottom = Inches(0.2)

    p = tf_lc2.paragraphs[0]
    p.space_after = Pt(2)
    r = p.add_run()
    r.text = "OPEN-SOURCE CODEBASE & PIPELINE"
    r.font.name = FONT_NAME
    r.font.size = Pt(9.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_GREEN

    p = tf_lc2.add_paragraph()
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = "GitHub Repository: AdityaG346/Sandisk"
    r.font.name = FONT_NAME
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = COLOR_WHITE

    p = tf_lc2.add_paragraph()
    r = p.add_run()
    r.text = "https://github.com/AdityaG346/Sandisk"
    r.font.name = FONT_NAME
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_GREEN
    r.hyperlink.address = "https://github.com/AdityaG346/Sandisk"

    closing = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.7), Inches(5.9), Inches(5.833), Inches(0.8))
    closing.fill.solid()
    closing.fill.fore_color.rgb = RGBColor(15, 23, 42)
    closing.line.color.rgb = COLOR_AMBER
    tf_cl = closing.text_frame
    tf_cl.word_wrap = True
    tf_cl.margin_left = tf_cl.margin_top = tf_cl.margin_right = tf_cl.margin_bottom = Inches(0.12)

    p = tf_cl.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Team Cookies of the Dark Web | SanDisk Cerebrum 2026\nThank you! Handing over to Speaker 4 for the Live Demo."
    r.font.name = FONT_NAME
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = COLOR_AMBER

    # Save presentation
    output_path = OUTPUT_DIR / "pitch_deck.pptx"
    prs.save(str(output_path))
    print(f"Successfully generated pitch deck at {output_path} with {len(prs.slides)} slides.")


if __name__ == "__main__":
    build_deck()
