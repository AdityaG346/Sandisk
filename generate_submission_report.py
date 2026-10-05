"""
generate_submission_report.py
=============================
Generates the official submission PDF report at outputs/submission_report.pdf
for SanDisk Cerebrum 2026 hackathon.

Assembles and formats content strictly from:
- docs/project_explanation.md
- outputs/results_summary.md
- outputs/comparison_table.csv

No new numbers, findings, or analyses are created.
"""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SHAP_DIR = OUTPUTS_DIR / "shap"
PDF_PATH = OUTPUTS_DIR / "submission_report.pdf"


# -----------------------------------------------------------------------------
# Numbered Canvas for Professional Two-Pass Header & Footer
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        page_w = 612.0
        page_h = 792.0
        margin_x = 36.0
        
        # 1. Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#0F172A"))
            self.drawString(margin_x, page_h - 26, "DIE YIELD PREDICTION — SANDISK CEREBRUM 2026")
            
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(page_w - margin_x, page_h - 26, "Team: Cookies of the Dark Web")
            
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.6)
            self.line(margin_x, page_h - 30, page_w - margin_x, page_h - 30)

        # 2. Running Footer (All Pages)
        footer_y = 22.0
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(margin_x, footer_y + 12, page_w - margin_x, footer_y + 12)

        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1D4ED8"))
        
        # GitHub Link Text
        gh_text = "GitHub: https://github.com/AdityaG346/Sandisk.git"
        self.drawString(margin_x, footer_y, gh_text)
        gh_w = self.stringWidth(gh_text, "Helvetica-Bold", 7.5)

        # Divider
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#94A3B8"))
        self.drawString(margin_x + gh_w + 6, footer_y, "|")

        # Dashboard Link Text
        dash_x = margin_x + gh_w + 14
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#0284C7"))
        dash_text = "Dashboard: https://die-yield-prediction.streamlit.app/"
        self.drawString(dash_x, footer_y, dash_text)
        dash_w = self.stringWidth(dash_text, "Helvetica-Bold", 7.5)

        # Page Number Right Aligned
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_w - margin_x, footer_y, page_str)


# -----------------------------------------------------------------------------
# PDF Builder Function
# -----------------------------------------------------------------------------
def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=38,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_blue = colors.HexColor("#1D4ED8")       # Blue 700
    c_sky = colors.HexColor("#0284C7")        # Sky 600
    c_dark = colors.HexColor("#1E293B")       # Slate 800
    c_muted = colors.HexColor("#475569")      # Slate 600
    c_bg_card = colors.HexColor("#F8FAFC")    # Slate 50
    c_border = colors.HexColor("#E2E8F0")     # Slate 200

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=c_primary,
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_blue,
        alignment=0,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_blue,
        spaceBefore=5,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=13.0,
        textColor=c_dark,
        spaceAfter=5,
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    caption_style = ParagraphStyle(
        "CaptionText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.5,
        textColor=c_muted,
        alignment=0,
    )

    link_box_title = ParagraphStyle(
        "LinkBoxTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=c_primary,
    )

    link_box_text = ParagraphStyle(
        "LinkBoxText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=c_blue,
    )

    story = []

    # =========================================================================
    # PAGE 1: Title Page
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("DIE YIELD PREDICTION", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("SanDisk Cerebrum 2026 Hackathon — SRM Institute of Science & Technology Project", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=c_blue, spaceBefore=2, spaceAfter=14))

    # Team & Project Metadata Box
    team_data = [
        [
            Paragraph("<b>Team Name:</b>", body_style),
            Paragraph("Cookies of the Dark Web", body_bold),
        ],
        [
            Paragraph("<b>Team Members:</b>", body_style),
            Paragraph("Shrey Sharma, Aditya Gupta, Prisha Kushwaha, Darsh Ramoliya", body_style),
        ],
        [
            Paragraph("<b>Core Problem:</b>", body_style),
            Paragraph("Identify which pre-test passing dies (<code>old_label=0</code>) fail post-burn-in testing (<code>label=1</code>)", body_style),
        ],
        [
            Paragraph("<b>Evaluated Models:</b>", body_style),
            Paragraph("Model A (500 Parametric + 10 Spatial + 1 Anomaly = 511) vs. Model B (+ 20 Sub-Die Block Features = 531)", body_style),
        ],
        [
            Paragraph("<b>Primary Metric:</b>", body_style),
            Paragraph("Post-test Fail F1 score on eligible dies (secondary metric: PR-AUC)", body_style),
        ],
    ]
    t_team = Table(team_data, colWidths=[110, 430])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 14))

    # Prominent Tagline from project_explanation.md
    tagline_p = Paragraph(
        "<b>Executive Tagline:</b><br/>"
        "<i>\"In semiconductor manufacturing, silicon chips, called dies, are fabricated by the thousands across large circular discs called wafers. "
        "Before chips are packaged into phones or computers, they undergo rigorous stress testing to filter out hidden defects. "
        "This project predicts which currently healthy-looking dies will fail post-stress testing before those tests even finish. "
        "By accurately spotting defective dies early, manufacturers can prevent faulty chips from shipping to customers, optimize testing schedules, "
        "and avoid wasting money packaging parts that are doomed to fail.\"</i>",
        body_style
    )
    t_tagline = Table([[tagline_p]], colWidths=[540])
    t_tagline.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_tagline)
    story.append(Spacer(1, 14))

    # Prominent Links & Resources Block
    story.append(Paragraph("Official Project Repositories & Live Resources", h2_style))
    links_data = [
        [
            Paragraph("<b>GitHub Repository:</b>", link_box_title),
            Paragraph("<a href='https://github.com/AdityaG346/Sandisk.git'><u>https://github.com/AdityaG346/Sandisk.git</u></a>", link_box_text)
        ],
        [
            Paragraph("<b>Live Cloud Dashboard:</b>", link_box_title),
            Paragraph("<a href='https://die-yield-prediction.streamlit.app/'><u>https://die-yield-prediction.streamlit.app/</u></a>", link_box_text)
        ],
        [
            Paragraph("<b>Full Reproducibility:</b>", link_box_title),
            Paragraph("Full code, model checkpoints, and reproduction instructions are in the GitHub repository README.", body_style)
        ]
    ]
    t_links = Table(links_data, colWidths=[150, 390])
    t_links.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_links)
    story.append(Spacer(1, 14))

    # Report Contents Map
    story.append(Paragraph("Submission Report Structure", h2_style))
    toc_data = [
        [Paragraph("<b>Page 2</b>", body_bold), Paragraph("Problem Summary & Factory Defect Reality", body_style)],
        [Paragraph("<b>Page 3</b>", body_bold), Paragraph("Model A & Model B Architectures and Feature Engineering", body_style)],
        [Paragraph("<b>Page 4</b>", body_bold), Paragraph("Key Experimental Results & Formal Statistical Significance", body_style)],
        [Paragraph("<b>Page 5</b>", body_bold), Paragraph("Interpretability Layer: Wafer Risk Fields & TreeSHAP Attribution", body_style)],
        [Paragraph("<b>Page 6</b>", body_bold), Paragraph("Severe Class Imbalance, Signal Overlap & Cost-Sensitive Tuning", body_style)],
        [Paragraph("<b>Page 7</b>", body_bold), Paragraph("Expected Project Deliverables Checklist (Deliverables 1–5)", body_style)],
        [Paragraph("<b>Page 8</b>", body_bold), Paragraph("Links, Resources & Submission File Verification", body_style)],
    ]
    t_toc = Table(toc_data, colWidths=[65, 475])
    t_toc.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_toc)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Problem Summary
    # =========================================================================
    story.append(Paragraph("1. Problem Summary & Manufacturing Reality", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph(
        "<b>What This Project Does</b> (from <code>docs/project_explanation.md</code>):",
        h2_style
    ))
    story.append(Paragraph(
        "In semiconductor manufacturing, silicon chips, called dies, are fabricated by the thousands across large circular discs called wafers. "
        "Before chips are packaged into phones or computers, they undergo rigorous stress testing to filter out hidden defects. "
        "However, physically testing every single die after stressful burn-in cycles is slow, expensive, and consumes factory bandwidth. "
        "This project predicts which currently healthy-looking dies will fail post-stress testing before those tests even finish. "
        "By accurately spotting defective dies early, manufacturers can prevent faulty chips from shipping to customers, optimize testing schedules, "
        "and avoid wasting money packaging parts that are doomed to fail.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("The Critical Eligibility Rule & True Defect Rates", h2_style))
    story.append(Paragraph(
        "A rigorous understanding of the data generation process reveals that dies fall into two distinct groups: those that already failed prior to burn-in (<code>old_label == 1</code>) "
        "and those that passed pre-test inspection (<code>old_label == 0</code>). The evaluation logic <b>strictly considers eligible dies only</b>. "
        "Dies that were already failed trivially remain failed; including them in the evaluation pool artificially inflates accuracy metrics and masks the model's true diagnostic utility. "
        "Direct verification from <code>input/train.csv</code> and <code>input/test.csv</code> confirms the exact population breakdown:",
        body_style
    ))
    story.append(Spacer(1, 4))

    data_breakdown = [
        [Paragraph("<b>Metric / Population Category</b>", body_bold), Paragraph("<b>Formula / Definition</b>", body_bold), Paragraph("<b>Training Set (160 Wafers)</b>", body_bold), Paragraph("<b>Test Set (40 Wafers)</b>", body_bold)],
        [Paragraph("Total Silicon Dies", body_style), Paragraph("Total dataset rows", body_style), Paragraph("173,099", body_style), Paragraph("39,351", body_style)],
        [Paragraph("Pre-test Defective Dies", body_style), Paragraph("<code>old_label == 1</code> (excluded)", body_style), Paragraph("19,062 (11.01%)", body_style), Paragraph("6,753 (17.16%)", body_style)],
        [Paragraph("<b>Eligible Candidate Dies</b>", body_bold), Paragraph("<code>old_label == 0</code> (evaluated)", body_style), Paragraph("<b>154,037 (88.99%)</b>", body_bold), Paragraph("<b>32,598 (82.84%)</b>", body_bold)],
        [Paragraph("<b>Eligible New Failures</b>", body_bold), Paragraph("<code>label == 1 & old_label == 0</code>", body_style), Paragraph("<b>6,519</b>", body_bold), Paragraph("<b>1,380</b>", body_bold)],
        [Paragraph("<b>True Eligible Fail Rate</b>", body_bold), Paragraph("Eligible Fails / Eligible Dies", body_style), Paragraph("<b>4.2321%</b>", body_bold), Paragraph("<b>4.2334%</b>", body_bold)],
        [Paragraph("Overall Positive Rate", body_style), Paragraph("All Fails / All Dies (conflated)", body_style), Paragraph("14.7783% (14.78%)", body_style), Paragraph("20.6678%", body_style)],
    ]
    t_breakdown = Table(data_breakdown, colWidths=[130, 150, 130, 130])
    t_breakdown.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_breakdown)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Autopsy of Prior Data Discrepancies (from <code>outputs/results_summary.md</code>)", h2_style))
    story.append(Paragraph(
        "• <b>The 14.78% vs. 4.23% Defect Rate Resolution</b>: An initial exploratory script reported a 14.78% fail rate by computing <code>train['label'].mean()</code> (25,581 / 173,099). "
        "This conflated pre-test dead dies with new failures. Among eligible dies (<code>old_label == 0</code>), exactly <b>4.2321%</b> fail in training and <b>4.2334%</b> fail in test, precisely matching the synthetic generator's true base defect rate.<br/>"
        "• <b>The 35,022 vs. 32,598 Test-Set Eligible Die Discrepancy</b>: Earlier documentation projected an 11.0% pre-test defect rate onto test wafers (39,351 − 4,329 = 35,022). "
        "Direct fresh count from <code>test.csv</code> reveals exactly <b>32,598 eligible dies</b> because holdout test set wafers include several heavily defective wafers (notably <code>W_F_0010</code> with 2,371 pre-test dead dies out of 4,096, a 57.89% pre-test defect rate), elevating the test-wide pre-test defect rate to 17.16%.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Model A & Model B Architectures
    # =========================================================================
    story.append(Paragraph("2. Model Architectures & Feature Engineering", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("<b>Model A (Baseline + Spatial + Die Anomaly)</b>", h2_style))
    story.append(Paragraph(
        "<i>(from docs/project_explanation.md Section 2)</i><br/>"
        "Model A evaluates each candidate die using two primary sources of information: hundreds of electrical sensor measurements taken on that specific chip, "
        "combined with its physical location on the silicon wafer. Chips are not independent islands; manufacturing flaws often cluster in specific physical regions "
        "or along the wafer edges due to chemical and thermal gradients during fabrication. Model A looks at whether a die sits close to known defective neighbors, "
        "its distance to the wafer boundary, and how far its electrical behavior deviates from typical healthy chips. "
        "It combines these clues into an initial estimated failure probability for every eligible die.",
        body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Model B (Full Diagnostic Model with Sub-Die Block Readings)</b>", h2_style))
    story.append(Paragraph(
        "<i>(from docs/project_explanation.md Section 3)</i><br/>"
        "Model B builds directly on Model A by adding deep internal diagnostic data from within the microchip itself. Each die contains thousands of microscopic memory blocks, "
        "and the factory collects roughly two thousand fine-grained voltage and resistance readings across these internal structures. "
        "While Model A only sees the chip from the outside as a single entity, Model B inspects internal summary patterns, including whether certain internal blocks are drifting, "
        "fluctuating abnormally, or showing localized electrical weakness. It uses this sub-die sensor data to detect subtle internal warning signs that might not yet show up on macro-level chip sensors.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Feature Hierarchy & Input Data Sources Feeding Each Model", h2_style))
    
    feat_data = [
        [Paragraph("<b>Feature Domain</b>", body_bold), Paragraph("<b>Features Included</b>", body_bold), Paragraph("<b>Description & Extraction Logic</b>", body_bold), Paragraph("<b>Model A</b>", body_bold), Paragraph("<b>Model B</b>", body_bold)],
        [
            Paragraph("<b>Parametric Electrical Tests</b>", body_bold),
            Paragraph("500 features<br/>(<code>feature_1</code> ... <code>feature_500</code>)", body_style),
            Paragraph("Macro-level continuous electrical and parametric measurements per die.", body_style),
            Paragraph("<b>YES</b> (500)", body_style),
            Paragraph("<b>YES</b> (500)", body_style),
        ],
        [
            Paragraph("<b>Leakage-Free Spatial Context</b>", body_bold),
            Paragraph("10 features<br/>(<code>sp_dist_to_fail</code>, <code>sp_edge_prox</code>, etc.)", body_style),
            Paragraph("Wafer geometry, radial distance, edge proximity, zone yield, and neighborhood defect counts derived <i>strictly from pre-test maps</i>.", body_style),
            Paragraph("<b>YES</b> (10)", body_style),
            Paragraph("<b>YES</b> (10)", body_style),
        ],
        [
            Paragraph("<b>Die Anomaly Detector</b>", body_bold),
            Paragraph("1 feature<br/>(<code>die_anomaly_score</code>)", body_style),
            Paragraph("Unsupervised Isolation Forest (100 trees, 5% contamination) fit exclusively on healthy passing dies (<code>old_label == 0</code>).", body_style),
            Paragraph("<b>YES</b> (1)", body_style),
            Paragraph("<b>YES</b> (1)", body_style),
        ],
        [
            Paragraph("<b>Sub-Die Block Telemetry</b>", body_bold),
            Paragraph("20 features<br/>(8 stats + 1 anomaly + 11 telemetry)", body_style),
            Paragraph("Statistical moments, tail percentiles, and MAD anomaly score summarizing ~2,000 internal block readings per die.", body_style),
            Paragraph("NO (0)", body_style),
            Paragraph("<b>YES</b> (20)", body_style),
        ],
        [
            Paragraph("<b>Total Feature Count</b>", body_bold),
            Paragraph("—", body_style),
            Paragraph("Gradient-boosted decision trees (LightGBM, <code>num_leaves=63</code>, <code>learning_rate=0.05</code>, <code>scale_pos_weight=12.0</code>).", body_style),
            Paragraph("<b>511 Features</b>", body_bold),
            Paragraph("<b>531 Features</b>", body_bold),
        ],
    ]
    t_feat = Table(feat_data, colWidths=[105, 95, 200, 70, 70])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#EFF6FF")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Leakage-Prevention Guardrails", h2_style))
    story.append(Paragraph(
        "To guarantee valid generalization, spatial defect statistics are computed strictly on pre-test dies (<code>old_label</code>) and never post-test labels (<code>label</code>). "
        "Every merge point in <code>src/</code> is protected by permanent automated index assertions (<code>assert_aligned</code>) asserting composite key equivalence (<code>wafer_id, die_row, die_col</code>). "
        "Wafer splits are verified disjoint to eliminate data leakage across manufacturing lots.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Key Results & Statistical Significance
    # =========================================================================
    story.append(Paragraph("3. Key Experimental Results & Statistical Rigor", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("<b>What We Found Comparing Them</b> (from <code>docs/project_explanation.md</code> Section 4):", h2_style))
    story.append(Paragraph(
        "When we rigorously compared both systems across thousands of test dies and multiple random splits, we discovered that adding internal block readings "
        "provided a small but statistically genuine improvement in ranking risk, yet made virtually no difference in the final pass-or-fail classification. "
        "In practical terms, Model B does a better job of ordering dies from most risky to least risky. However, because factory defects in this manufacturing "
        "process are deliberately subtle and share nearly identical sensor values with healthy chips, the block signals are too faint to convert borderline calls "
        "into confident failure calls at a fixed decision threshold.",
        body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Head-to-Head Performance Benchmark (from <code>outputs/comparison_table.csv</code>)", h2_style))
    
    comp_data = [
        [Paragraph("<b>Metric</b>", body_bold), Paragraph("<b>Model A</b>", body_bold), Paragraph("<b>Model B</b>", body_bold), Paragraph("<b>Delta (B − A)</b>", body_bold), Paragraph("<b>Relative Delta</b>", body_bold), Paragraph("<b>Status / Rigor</b>", body_bold)],
        [Paragraph("<b>Fail F1 Score</b> (Primary)", body_bold), Paragraph("0.5209 (0.5207)", body_style), Paragraph("0.5217 (0.5222)", body_style), Paragraph("<b>+0.0008 (+0.0015)</b>", body_bold), Paragraph("+0.15% to +0.29%", body_style), Paragraph("<font color='#B45309'><b>NOT Significant (Noise)</b></font>", body_style)],
        [Paragraph("<b>PR-AUC</b> (Ranking)", body_bold), Paragraph("0.5014 (0.5024)", body_style), Paragraph("0.5357 (0.5361)", body_style), Paragraph("<b>+0.0343 (+0.0337)</b>", body_bold), Paragraph("+6.71% to +6.84%", body_style), Paragraph("<font color='#166534'><b>SIGNIFICANT (p &lt; 0.001)</b></font>", body_style)],
        [Paragraph("<b>Fail Recall</b>", body_bold), Paragraph("35.65% (492 / 1,380)", body_style), Paragraph("37.03% (511 / 1,380)", body_style), Paragraph("+1.38% (+19 dies)", body_style), Paragraph("+3.87%", body_style), Paragraph("<font color='#166534'><b>SIGNIFICANT (p &lt; 0.01)</b></font>", body_style)],
        [Paragraph("<b>Fail Precision</b>", body_bold), Paragraph("96.66% (492 / 509)", body_style), Paragraph("88.26% (511 / 579)", body_style), Paragraph("−8.40% (+51 FP)", body_style), Paragraph("−8.69%", body_style), Paragraph("High Precision Regime", body_style)],
        [Paragraph("<b>Overall Accuracy</b>", body_bold), Paragraph("97.22% (31,693 / 32,598)", body_style), Paragraph("97.13% (31,661 / 32,598)", body_style), Paragraph("−0.09%", body_style), Paragraph("−0.09%", body_style), Paragraph("Pass Majority Dominated", body_style)],
        [Paragraph("<b>Optimal Threshold</b>", body_bold), Paragraph("0.5574", body_style), Paragraph("0.5180", body_style), Paragraph("−0.0394", body_style), Paragraph("Independent Sweep", body_style), Paragraph("Calibrated per Model", body_style)],
    ]
    t_comp = Table(comp_data, colWidths=[115, 95, 95, 85, 75, 75])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Formal Hypothesis Testing & Bootstrap Verification", h2_style))
    story.append(Paragraph(
        "To rigorously confirm whether the observed differences reflect true architectural gains rather than random split noise, we performed both 5-seed validation hypothesis testing and a 1,000-sample test set bootstrap (from <code>outputs/results_summary.md</code> Section 3.B/C):",
        body_style
    ))
    story.append(Spacer(1, 3))

    stat_data = [
        [Paragraph("<b>Metric</b>", body_bold), Paragraph("<b>5-Seed Paired Delta (B − A)</b>", body_bold), Paragraph("<b>Paired t-test</b>", body_bold), Paragraph("<b>1,000-Sample Test Bootstrap 95% CI</b>", body_bold), Paragraph("<b>Statistically Significant?</b>", body_bold)],
        [
            Paragraph("<b>PR-AUC</b>", body_bold),
            Paragraph("+0.0359 ± 0.0031", body_style),
            Paragraph("t = +25.83<br/><b>p = 1.33 &times; 10<sup>-5</sup></b>", body_style),
            Paragraph("<b>[+0.022261, +0.046552]</b><br/>(Mean: +0.034434, excludes 0)", body_style),
            Paragraph("<font color='#166534'><b>YES (p &lt; 0.001)</b><br/>Definite Ranking Gain</font>", body_style),
        ],
        [
            Paragraph("<b>Fail F1 Score</b>", body_bold),
            Paragraph("+0.0090 ± 0.0098", body_style),
            Paragraph("t = +2.06<br/><b>p = 0.1081</b>", body_style),
            Paragraph("<b>[−0.008743, +0.010780]</b><br/>(Mean: +0.001304, contains 0)", body_style),
            Paragraph("<font color='#B45309'><b>NO (p &gt; 0.05)</b><br/>Within Random Noise</font>", body_style),
        ],
    ]
    t_stat = Table(stat_data, colWidths=[80, 115, 95, 135, 115])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Explicit Statistical Conclusion:</b><br/>"
        "• <b>The PR-AUC gain (+0.034 to +0.036) is statistically significant (<i>p</i> &lt; 0.001, 95% CI strictly excludes zero)</b>. Internal block readings provide smooth secondary continuous gradients that significantly improve the model's ability to rank high-risk dies across all probability thresholds.<br/>"
        "• <b>The Fail F1 gain (+0.0008 to +0.0015) is NOT statistically significant (<i>p</i> = 0.1081, 95% CI contains zero)</b>. At a discrete binary decision cutoff, 65% of new failures are marginal defects whose block readings overlap heavily with passing chips, preventing block features from converting borderline dies into confident failure classifications.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: Interpretability Layer
    # =========================================================================
    story.append(Paragraph("4. Interpretability, SHAP Attribution & Spatial Risk Fields", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=6))

    story.append(Paragraph("Continuous 2D Gaussian Wafer Risk Field with Hotspot Contours", h2_style))
    
    # Embed Risk Field Heatmap
    rf_img_path = str(PLOTS_DIR / "risk_field_wafer_W_F_0014.png")
    if os.path.exists(rf_img_path):
        story.append(Image(rf_img_path, width=530, height=125))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<i>Caption: Visualizes how raw discrete die failure probabilities smooth into continuous thermal risk fields (σ = 1.5) that accurately encircle true defect cluster geometries with 90th-percentile hotspot contour overlays for wafer W_F_0014 (from outputs/results_summary.md Section 7.C).</i>",
        caption_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("TreeSHAP Feature Attribution & Failure Signatures", h2_style))

    # Two-column layout: Left = SHAP summary image, Right = Domain Mass Breakdown + Signatures
    shap_img_path = str(SHAP_DIR / "shap_summary_model_b.png")
    left_cell = []
    if os.path.exists(shap_img_path):
        left_cell.append(Image(shap_img_path, width=175, height=270))
        left_cell.append(Spacer(1, 2))
        left_cell.append(Paragraph("<i>Model B TreeSHAP Summary (Top 25)</i>", caption_style))

    right_cell = [
        Paragraph(
            "<b>TreeSHAP Attribution Mass Breakdown:</b><br/>"
            "TreeSHAP analysis on the 500-tree Model B reveals the exact contribution of each physical domain (from <code>outputs/results_summary.md</code> Section 4.A):",
            body_style
        ),
        Spacer(1, 3),
    ]

    mass_data = [
        [Paragraph("<b>Domain</b>", body_bold), Paragraph("<b>Features</b>", body_bold), Paragraph("<b>Attribution Mass</b>", body_bold)],
        [Paragraph("Parametric & Anomaly", body_style), Paragraph("501", body_style), Paragraph("<b>84.48%</b> (12.86)", body_bold)],
        [Paragraph("Spatial Proximity", body_style), Paragraph("24", body_style), Paragraph("<b>10.34%</b> (1.57)", body_bold)],
        [Paragraph("Sub-Die Block Readings", body_style), Paragraph("9", body_style), Paragraph("<b>5.19%</b> (0.79)", body_bold)],
    ]
    t_mass = Table(mass_data, colWidths=[120, 60, 140])
    t_mass.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    right_cell.append(t_mass)
    right_cell.append(Spacer(1, 5))

    right_cell.append(Paragraph(
        "<b>Top Individual Features:</b><br/>"
        "1. <code>sp_dist_to_fail</code> (Spatial): 1.025 mean |SHAP| (6.73%)<br/>"
        "2. <code>blk_mean</code> (Block): 0.3806 mean |SHAP| (2.50%)<br/>"
        "3. <code>sp_old_label</code> (Spatial): 0.1624 mean |SHAP| (1.07%)<br/>"
        "4. <code>blk_q75</code> (Block): 0.1298 mean |SHAP| (0.85%)",
        body_style
    ))
    right_cell.append(Spacer(1, 5))

    right_cell.append(Paragraph(
        "<b>Discovered Failure Signatures (HDBSCAN):</b><br/>"
        "• <b>Cluster 0 (14 dies) — Memory Array Reading Shift</b>: Dominated by <code>blk_mean</code> (1.051) and <code>blk_q75</code> (0.362).<br/>"
        "• <b>Cluster 1 (11 dies) — Extreme Block Anomaly</b>: Uniform variance drift.<br/>"
        "• <b>Cluster 2 (373 dies) — Defect Proximity</b>: Dominated by <code>sp_dist_to_fail</code> (1.025).<br/>"
        "• <b>Cluster 3 (17 dies) — Spatial Defect & Edge Drift</b>.",
        body_style
    ))

    t_interp_split = Table([[left_cell, right_cell]], colWidths=[185, 345])
    t_interp_split.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_interp_split)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<i>Caption: TreeSHAP attribution on the fully converged 500-tree Model B demonstrates that die parametric & anomaly features drive 84.48% of attribution mass, spatial proximity drives 10.34% (dominated by sp_dist_to_fail at 1.025), and sub-die block readings contribute 5.19% (led by blk_mean at 0.3806) (from outputs/results_summary.md Section 4.A/B).</i>",
        caption_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: Imbalance & Overlap Handling
    # =========================================================================
    story.append(Paragraph("5. Class Imbalance, Signal Overlap & Production Tuning", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph(
        "This section reproduces the four foundational principles of class imbalance and signal overlap management from <code>outputs/results_summary.md</code> Section 6:",
        body_style
    ))
    story.append(Spacer(1, 4))

    imb_points = [
        [
            Paragraph("<b>1. The Class Distribution Challenge</b>", h2_style),
            Paragraph(
                "The prediction task operates under severe class imbalance and intentional distribution overlap. "
                "As verified in Section 1, eligible dies (<code>old_label == 0</code>) exhibit a post-test failure rate of only <b>4.23%</b> "
                "(a 1 : 22.6 positive-to-negative ratio; 6,519 fails out of 154,037 train dies, and 1,380 fails out of 32,598 test dies). "
                "Compounding this scarcity, <b>65% of all new failures are synthetically designated as 'marginal'</b> by <code>generate_data.py</code>, "
                "meaning their parametric measurements are drawn from distributions with low signal-to-noise separation, making them near-indistinguishable "
                "from the pass population by design.",
                body_style
            )
        ],
        [
            Paragraph("<b>2. Class-Weighting Strategy & Rationale for scale_pos_weight = 12.0</b>", h2_style),
            Paragraph(
                "Although the theoretical inverse ratio for a 4.23% positive rate is (1 − 0.0423) / 0.0423 ≈ 22.63, our grid search across "
                "<code>[3.0 ... 40.0]</code> (Section 5) proved that weights ≥ 17.0 cause violent gradient oscillations, causing early stopping to abort "
                "prematurely after 2 to 8 trees. Between the viable stable weights, <code>5.76</code> nominally peaked in validation F1 (0.5269), while "
                "<code>12.0</code> achieved <b>0.5254 F1 with peak precision (93.37% validation, and 88.26%–96.66% test)</b>. "
                "In silicon fabrication, false alarms trigger costly scrap of functional dies; <code>scale_pos_weight = 12.0</code> was selected because it maximizes "
                "precision against the 95.77% healthy majority while sacrificing less than 0.15% in F1.",
                body_style
            )
        ],
        [
            Paragraph("<b>3. Empirical Refutation of the 'Predict All Pass' Shortcut</b>", h2_style),
            Paragraph(
                "A naive classifier predicting the dominant class ('all pass') would trivially achieve 95.77% accuracy on eligible dies, but would yield "
                "<b>0.00% Fail Recall</b> and an F1 of 0.0000. In sharp contrast, our test set evaluation (Section 3.A) demonstrates "
                "<b>Fail Recall of 35.65% (Model A) and 37.03% (Model B)</b> alongside <b>Fail Precision of 96.66% (Model A) and 88.26% (Model B)</b>, "
                "successfully capturing 492 to 511 true defective dies with only 17 to 68 false positives out of 31,218 healthy dies.",
                body_style
            )
        ],
        [
            Paragraph("<b>4. Necessity of Post-Hoc Threshold Tuning</b>", h2_style),
            Paragraph(
                "Because individual parametric features exhibit minimal univariate effect sizes (Cohen's <i>d</i> ≈ 0.09–0.23 across <code>feature_1</code>...<code>feature_500</code>), "
                "class weighting alone cannot disentangle overlapping class densities; explicit post-hoc threshold calibration (tuning independently to "
                "<b>0.5180 for Model B</b> and <b>0.5574 for Model A</b> on validation F1) was necessary to position the decision boundary past the dense pass-population margin.",
                body_style
            )
        ],
    ]

    for p_title, p_content in imb_points:
        t_box = Table([[p_title], [p_content]], colWidths=[540])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
            ('LINEBEFORE', (0,0), (0,-1), 2.5, c_blue),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_box)
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: Expected Deliverables Checklist
    # =========================================================================
    story.append(Paragraph("6. Expected Project Deliverables Checklist", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph(
        "All five required project deliverables were built, tested, audited, and committed to the repository. "
        "The paragraphs below are pulled directly from <code>docs/project_explanation.md</code>:",
        body_style
    ))
    story.append(Spacer(1, 4))

    deliv_items = [
        (
            "Deliverable 1: Modular Source Code Library (src/)",
            "We built a clean, modular Python codebase organized within the source directory. This package contains dedicated modules for loading wafer datasets, "
            "engineering leakage-free spatial features, extracting block statistics, fitting healthy-chip anomaly detectors, training and predicting with both models, "
            "running rigorous performance evaluations, and generating publication-quality wafer visualizations. Every data join in these modules is protected by permanent "
            "automated alignment assertions that prevent silent data-slicing errors. The entire codebase is organized into reusable components and can be found in the source directory (<code>src/</code>)."
        ),
        (
            "Deliverable 2: Reproducible End-to-End Master Pipeline (run_all.py)",
            "We created a fully automated master pipeline script that reproduces the entire project workflow from start to finish with a single command. "
            "The script loads the raw wafer datasets, runs exploratory data analysis, computes spatial and block features, trains all baseline and advanced models, "
            "selects optimal decision thresholds on validation splits, evaluates metrics on holdout test wafers, and generates all visual maps and diagnostic plots. "
            "Anyone can regenerate the entire benchmark and all accompanying charts from scratch by running the master pipeline script (<code>run_all.py</code>)."
        ),
        (
            "Deliverable 3: Trained Production Models and Calibrated Decision Rules",
            "We produced and saved the finalized, trained machine learning model files for both Model A and Model B, alongside their standalone threshold configuration metadata. "
            "Each model file contains the optimized decision tree ensemble ready for immediate deployment in an automated test environment. Because raw probabilities require careful "
            "calibration against high-volume factory scrap costs, each model includes its own independently tuned decision cutoff designed to catch defective chips while preserving high precision. "
            "The serialized model files and their metadata are stored in the outputs directory (<code>outputs/model_a.pkl</code>, <code>outputs/model_b.pkl</code>, and accompanying metadata files)."
        ),
        (
            "Deliverable 4: Model Evaluation and Feature Ablation Tables",
            "We generated standardized performance tables comparing Model A and Model B across all industry-standard metrics, including failure recall, failure precision, "
            "balanced accuracy, and overall classification accuracy. To ensure scientific rigor, these tables include five-seed multi-split variance estimates, "
            "formal statistical significance hypothesis tests, and a one-thousand-run bootstrap confidence interval on holdout wafers. We also produced feature ablation tables "
            "showing the exact incremental value of die sensors, spatial geography, and block readings. All summary evaluation spreadsheets are saved as clean comma-separated files "
            "in the outputs directory (<code>outputs/comparison_table.csv</code>, <code>outputs/ablation_table_multiseed.csv</code>, and <code>outputs/multiseed_statistical_tests.csv</code>)."
        ),
        (
            "Deliverable 5: Comprehensive Technical Summary and Diagnostic Report",
            "We authored an exhaustive, end-to-end technical results report detailing all exploratory data findings, data rate verifications, engineering autopsies of past software bugs, "
            "multi-seed statistical significance results, and physical feature attribution percentages. Alongside this report, we generated a catalog of standardized per-die explanations "
            "with ensemble confidence intervals and four-panel visual wafer maps featuring smoothed defect heat surfaces with automated hotspot boundaries. The full written summary report, "
            "along with individual die diagnostic files and wafer heatmaps, is located in the outputs directory (<code>outputs/results_summary.md</code>, <code>outputs/per_die_explanations.md</code>, and <code>outputs/plots/</code>)."
        ),
    ]

    for d_title, d_text in deliv_items:
        t_d = Table([[Paragraph(f"<b>{d_title}</b>", h2_style)], [Paragraph(d_text, body_style)]], colWidths=[540])
        t_d.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('LINEBEFORE', (0,0), (0,-1), 2.5, colors.HexColor("#10B981")),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_d)
        story.append(Spacer(1, 4.5))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: Links, Resources & Submission Verification
    # =========================================================================
    story.append(Paragraph("7. Repository Links, Live Dashboard & Verification", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph(
        "<b>Prominent Links & Resources</b> (repeat from Title Page for instant access):",
        h2_style
    ))
    story.append(Spacer(1, 4))

    # Prominent large links block
    p8_links_data = [
        [
            Paragraph("<b>GitHub Repository:</b>", link_box_title),
            Paragraph("<font size=10><a href='https://github.com/AdityaG346/Sandisk.git'><b><u>https://github.com/AdityaG346/Sandisk.git</u></b></a></font>", link_box_text),
        ],
        [
            Paragraph("<b>Live Cloud Dashboard:</b>", link_box_title),
            Paragraph("<font size=10><a href='https://die-yield-prediction.streamlit.app/'><b><u>https://die-yield-prediction.streamlit.app/</u></b></a></font>", link_box_text),
        ],
        [
            Paragraph("<b>Reproducibility Note:</b>", link_box_title),
            Paragraph("<b>\"Full code, models, and reproducibility instructions are in the GitHub repository README.\"</b>", body_bold),
        ],
    ]
    t_p8_links = Table(p8_links_data, colWidths=[150, 390])
    t_p8_links.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#2563EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.75, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_p8_links)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Submission Artifacts & Verification Checklist", h2_style))
    story.append(Paragraph(
        "Every file required for grading and automated re-scoring is verified and committed to the repository:",
        body_style
    ))
    story.append(Spacer(1, 4))

    artifacts_data = [
        [Paragraph("<b>Artifact / Deliverable</b>", body_bold), Paragraph("<b>File Location</b>", body_bold), Paragraph("<b>Verification Status & Schema</b>", body_bold)],
        [
            Paragraph("<b>Final Predictions CSV</b>", body_bold),
            Paragraph("<code>outputs/predictions.csv</code>", body_style),
            Paragraph("<b>39,351 dies</b> scored with Model B tuned cutoff (0.5180). Strict trivial-fail rule applied (all <code>old_label=1</code> set to 1). Schema matches <code>wafer_id, die_row, die_col, predicted_label</code>.", body_style),
        ],
        [
            Paragraph("<b>Production Model Checkpoints</b>", body_bold),
            Paragraph("<code>outputs/model_a.pkl</code><br/><code>outputs/model_b.pkl</code>", body_style),
            Paragraph("Fully converged 500-tree LightGBM ensembles with standalone serialized metadata (<code>model_a_meta.pkl</code>, <code>model_b_meta.pkl</code>).", body_style),
        ],
        [
            Paragraph("<b>Standardized Benchmark Tables</b>", body_bold),
            Paragraph("<code>outputs/comparison_table.csv</code><br/><code>outputs/ablation_table_multiseed.csv</code>", body_style),
            Paragraph("Full point-estimate comparisons, 5-seed validation ablations, and paired t-test / Wilcoxon significance statistics.", body_style),
        ],
        [
            Paragraph("<b>Comprehensive Technical Audit</b>", body_bold),
            Paragraph("<code>outputs/results_summary.md</code>", body_style),
            Paragraph("Exhaustive written report documenting fail-rate discrepancy autopsies, bug autopsies, class-weight curves, and TreeSHAP feature mass breakdown.", body_style),
        ],
        [
            Paragraph("<b>Automated Regression Suite</b>", body_bold),
            Paragraph("<code>test_part1_guardrails.py</code>", body_style),
            Paragraph("<b>5/5 Checks Passed</b>: Alignment assertions, training sanity floor, threshold independence, determinism, and disk metric reproduction.", body_style),
        ],
        [
            Paragraph("<b>Interactive Presentation Layer</b>", body_bold),
            Paragraph("<code>dashboard/app.py</code><br/><code>dashboard/requirements.txt</code>", body_style),
            Paragraph("Deployed on Streamlit Community Cloud. Features 4-panel wafer risk fields, side-by-side Model A vs B maps, and sub-die block strip plots.", body_style),
        ],
    ]
    t_artifacts = Table(artifacts_data, colWidths=[140, 160, 240])
    t_artifacts.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_artifacts)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Final Submission Summary", h2_style))
    story.append(Paragraph(
        "This project conclusively demonstrates that incorporating sub-die block readings into gradient-boosted decision trees delivers a <b>statistically significant improvement in continuous probability ranking (+0.034 PR-AUC, p &lt; 0.001)</b>, "
        "while binary decision accuracy (Fail F1) remains governed by macro-spatial defect clustering due to heavy overlap among marginal manufacturing flaws. "
        "By coupling high-precision cost-sensitive class weighting (<code>scale_pos_weight = 12.0</code>) with independent post-hoc threshold tuning, the system captures 37.03% of true new failures with 88.26%–96.66% precision, "
        "providing semiconductor manufacturers with a production-ready, interpretable yield-optimization engine.",
        body_style
    ))

    # Build the PDF using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
    file_size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
