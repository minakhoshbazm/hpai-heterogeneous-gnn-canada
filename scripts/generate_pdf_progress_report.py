import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_pdf_report(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1A365D")    # Dark Navy
    SECONDARY = colors.HexColor("#2B6CB0")  # Blue Accent
    TEXT_DARK = colors.HexColor("#2D3748")  # Charcoal
    BG_LIGHT = colors.HexColor("#F7FAFC")   # Light Grey
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        alignment=TA_LEFT,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        alignment=TA_LEFT,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK,
        alignment=TA_LEFT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title Block
    story.append(Paragraph("HPAI Heterogeneous Spatiotemporal GNN Canada", title_style))
    story.append(Paragraph("Comprehensive Project Progress, Modeling Benchmarks & Research Decision Record — August 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Core Objective", h1_style))
    story.append(Paragraph(
        "This project implements an automated, reproducible machine learning pipeline using <b>heterogeneous spatiotemporal graph neural networks (GNNs)</b> "
        "to forecast one-month-ahead Highly Pathogenic Avian Influenza (HPAI) poultry outbreak risk across Canadian regions. "
        "The primary prediction target is binary: <i>Will at least one poultry outbreak occur in Census Subdivision (CSD) r during month t+1, using only data available up to month t?</i>",
        body_style
    ))
    story.append(Paragraph("<b>Key Project Specifications:</b>", body_style))
    story.append(Paragraph("• <b>Spatial Unit:</b> Statistics Canada Census Subdivisions (CSDs) — 5,161 total CSD nodes (5,054 provincial CSDs used in modeling).", bullet_style))
    story.append(Paragraph("• <b>Temporal Coverage:</b> Monthly steps from 2022 to 2025 (2022–2024 for training/validation, 2025 strictly held out for testing).", bullet_style))
    story.append(Paragraph("• <b>Extreme Class Imbalance:</b> Outbreak positive rate is ~0.16% (79 positive CSD-months out of 55,594 test observations in 2025; ratio ~622:1).", bullet_style))

    # Feature Engineering
    story.append(Paragraph("2. Data Pipeline & Multi-Modal Feature Extraction", h1_style))
    story.append(Paragraph(
        "A multi-modal geospatial data ingestion and processing pipeline has been established, uniting agricultural census metrics, road infrastructure, satellite landcover, and climate covariates into a unified modeling table:",
        body_style
    ))
    story.append(Paragraph("• <b>Outbreak Labels:</b> CFIA outbreak records spatially assigned to CSD polygons and aggregated to monthly binary flags.", bullet_style))
    story.append(Paragraph("• <b>Poultry Agricultural Proxies:</b> Broiler, turkey, and poultry farm counts from the Census of Agriculture (CCS level lookup).", bullet_style))
    story.append(Paragraph("• <b>GLW Poultry Density:</b> Global Livestock Watch 2020 chicken and 2015 duck density summarized per CSD (mean, max, 95th percentile).", bullet_style))
    story.append(Paragraph("• <b>Road Infrastructure:</b> National Road Network (NRN) total length, road density, and distance to major highways per CSD.", bullet_style))
    story.append(Paragraph("• <b>GEE Dynamic World Landcover:</b> Monthly Dynamic World water, flooded vegetation, crops, and trees surface area metrics.", bullet_style))
    story.append(Paragraph("• <b>ERA5-Land Climate Pipeline:</b> Monthly temperature, total precipitation, and soil moisture anomalies extracted per CSD.", bullet_style))

    # Graph Architecture
    story.append(Paragraph("3. Graph Engineering & Adjacency Topology", h1_style))
    story.append(Paragraph(
        "To capture spatial diffusion and environmental connectivity between geographic units, multiple graph topology variants were built:",
        body_style
    ))
    story.append(Paragraph("• <b>Spatial Contiguity Graphs:</b> Rook (shared boundaries), Queen (shared corners), and Nearest-Neighbor Fallback (for island CSDs).", bullet_style))
    story.append(Paragraph("• <b>kNN-3 Spatial Graph:</b> Distance-based k-nearest neighbor network (k=3, distance threshold ≤ 100 km).", bullet_style))
    story.append(Paragraph("• <b>Snapshot-Specific Multi-Relation Graphs (RGCN):</b> Incorporates geographic distance, commercial poultry similarity (cosine similarity ≥ 0.70), and expanding-window water habitat similarity constructed strictly from historical observations up to month t.", bullet_style))

    # Benchmarking Results Table
    story.append(Spacer(1, 4))
    story.append(Paragraph("4. Benchmark Modeling Performance (Strict 2025 Test Split)", h1_style))
    story.append(Paragraph(
        "All models were evaluated on the unchanged 2025 held-out test split (N = 55,594 CSD-month observations, N_pos = 79). Performance is measured primarily via <b>PR-AUC</b> and top-K precision/recall due to extreme label imbalance:",
        body_style
    ))

    table_data = [
        [
            Paragraph("Model Variant", table_header_style),
            Paragraph("Graph / Model Architecture", table_header_style),
            Paragraph("PR-AUC", table_header_style),
            Paragraph("ROC-AUC", table_header_style),
            Paragraph("Prec@50", table_header_style),
            Paragraph("Rec@50", table_header_style),
            Paragraph("Status", table_header_style)
        ],
        [
            Paragraph("kNN-3 GraphSAGE (3-Seed)", table_cell_bold),
            Paragraph("kNN-3 Spatial Graph (k=3, d≤100km)", table_cell_style),
            Paragraph("0.0791", table_cell_bold),
            Paragraph("0.9168", table_cell_style),
            Paragraph("0.2000", table_cell_bold),
            Paragraph("0.1266", table_cell_bold),
            Paragraph("Valid Spatial Baseline", table_cell_style)
        ],
        [
            Paragraph("Relational GNN (3-Seed)", table_cell_bold),
            Paragraph("Snapshot Time-Safe Multi-Relation Graph", table_cell_style),
            Paragraph("0.0698", table_cell_style),
            Paragraph("0.9126", table_cell_style),
            Paragraph("0.1600", table_cell_style),
            Paragraph("0.1013", table_cell_style),
            Paragraph("Primary Valid RGCN", table_cell_style)
        ],
        [
            Paragraph("Strict Env + Lag LR", table_cell_bold),
            Paragraph("Tabular Linear Baseline (No Graph)", table_cell_style),
            Paragraph("0.0654", table_cell_style),
            Paragraph("0.9084", table_cell_style),
            Paragraph("0.1200", table_cell_style),
            Paragraph("0.0759", table_cell_style),
            Paragraph("Linear Baseline", table_cell_style)
        ],
        [
            Paragraph("Strict MLP (No Graph)", table_cell_bold),
            Paragraph("Tabular Neural Network", table_cell_style),
            Paragraph("0.0601", table_cell_style),
            Paragraph("0.9048", table_cell_style),
            Paragraph("0.1200", table_cell_style),
            Paragraph("0.0759", table_cell_style),
            Paragraph("Non-Graph Neural Net", table_cell_style)
        ],
        [
            Paragraph("Time-Persistence Baseline", table_cell_bold),
            Paragraph("Historical Outbreak Recurrence (2022-2024)", table_cell_style),
            Paragraph("0.0405", table_cell_style),
            Paragraph("0.7311", table_cell_style),
            Paragraph("0.2000", table_cell_bold),
            Paragraph("0.1270", table_cell_style),
            Paragraph("Recurrence Baseline", table_cell_style)
        ],
        [
            Paragraph("Superseded RGCN (Static)", table_cell_style),
            Paragraph("Static 2022-2024 Water Habitat Graph", table_cell_style),
            Paragraph("0.0838*", table_cell_style),
            Paragraph("0.9174", table_cell_style),
            Paragraph("0.2000", table_cell_style),
            Paragraph("0.1266", table_cell_style),
            Paragraph("SUPERSEDED (Leakage)", table_cell_style)
        ],
        [
            Paragraph("Spatiotemporal Directed RGCN", table_cell_bold),
            Paragraph("Directed forward temporal edges (t -> t+1)", table_cell_style),
            Paragraph("0.0510", table_cell_style),
            Paragraph("0.9177", table_cell_style),
            Paragraph("0.1600", table_cell_style),
            Paragraph("0.1013", table_cell_style),
            Paragraph("Stage 4 Baseline", table_cell_style)
        ],
        [
            Paragraph("Spatiotemporal Directed Transfer", table_cell_bold),
            Paragraph("Directed GNN + Pretrained Wild-Bird encoder", table_cell_style),
            Paragraph("0.0655", table_cell_bold),
            Paragraph("0.9126", table_cell_style),
            Paragraph("0.1200", table_cell_style),
            Paragraph("0.0759", table_cell_style),
            Paragraph("Stage 4 Transfer", table_cell_style)
        ]
    ]

    t = Table(table_data, colWidths=[105, 145, 50, 50, 45, 45, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0, 1), (-1, 1), BG_LIGHT),
        ('BACKGROUND', (0, 3), (-1, 3), BG_LIGHT),
        ('BACKGROUND', (0, 5), (-1, 5), BG_LIGHT),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # Rolling-origin transfer validation
    story.append(Paragraph("5. Rolling-Origin Supervised Transfer Results", h1_style))
    story.append(Paragraph(
        "The latest Stage 3 run evaluates a supervised transfer GraphSAGE model under strict rolling-origin folds, "
        "pretraining on wild-bird HPAI next-month presence before poultry outbreak fine-tuning. Pooled metrics combine "
        "the 2023, 2024, and 2025 out-of-sample test years (176,890 CSD-month rows; 191 positives).",
        body_style
    ))

    rolling_table_data = [
        [
            Paragraph("Model", table_header_style),
            Paragraph("Validation Window", table_header_style),
            Paragraph("PR-AUC", table_header_style),
            Paragraph("ROC-AUC", table_header_style),
            Paragraph("P@50", table_header_style),
            Paragraph("P@100", table_header_style),
            Paragraph("Bootstrap Decision", table_header_style)
        ],
        [
            Paragraph("Supervised Transfer GraphSAGE", table_cell_bold),
            Paragraph("Wild-bird pretrain -> poultry fine-tune", table_cell_style),
            Paragraph("0.0720", table_cell_bold),
            Paragraph("0.9054", table_cell_style),
            Paragraph("0.300", table_cell_bold),
            Paragraph("0.210", table_cell_bold),
            Paragraph("Better than LR; comparable/better than rolling GraphSAGE", table_cell_style)
        ],
        [
            Paragraph("Monthly-Snapshot GraphSAGE", table_cell_style),
            Paragraph("Rolling baseline", table_cell_style),
            Paragraph("0.0628", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("Transfer diff +0.0092; 83.9% win rate", table_cell_style)
        ],
        [
            Paragraph("Wild-Bird Logistic Regression", table_cell_style),
            Paragraph("Rolling tabular baseline", table_cell_style),
            Paragraph("0.0540", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("Transfer diff +0.0180; 98.6% win rate", table_cell_style)
        ],
    ]

    rolling_t = Table(rolling_table_data, colWidths=[118, 112, 48, 48, 42, 42, 120])
    rolling_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0, 1), (-1, 1), BG_LIGHT),
        ('BACKGROUND', (0, 3), (-1, 3), BG_LIGHT),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(rolling_t)
    story.append(Spacer(1, 6))

    # Research Decisions & Audit
    story.append(Paragraph("6. Critical Methodological Decisions & Audit Findings", h1_style))
    story.append(Paragraph("• <b>Time-Safety & Covariate Leakage Mitigation:</b> Replaced static graph construction with a <i>snapshot-specific time-safe graph pipeline</i>, establishing strict temporal isolation.", bullet_style))
    story.append(Paragraph("• <b>Supervised Transfer Learning (Stage 3):</b> Implemented transfer learning by pretraining a GraphSAGE encoder on wild-bird HPAI next-month presence (7,234 positives), then fine-tuning on poultry outbreaks (381 positives). This successfully leverages the 19x denser wild-bird signal.", bullet_style))
    story.append(Paragraph("• <b>Rolling-Origin Temporal Validation Results:</b> Under strict temporal folds (evaluating 2023, 2024, and 2025 out-of-sample), the Supervised Transfer GNN achieved a pooled PR-AUC of <b>0.0720</b>, outperforming the rolling baseline GraphSAGE (0.0628) and Wild-Bird Logistic Regression (0.0540). Bootstrap win rate vs. Logistic Regression is <b>98.6%</b> (statistically better).", bullet_style))

    # Roadmap
    story.append(Paragraph("7. Current Work & Next-Generation Architecture Roadmap", h1_style))
    story.append(Paragraph("• <b>Stage 4 Implementation:</b> Further refinement of the temporal transfer learning models, investigating directed temporal connections to model infection waves across subdivisions without leakage.", bullet_style))
    story.append(Paragraph("• <b>Data Sources:</b> Validated CFIA wild-bird detection data joined via robust geospatial crosswalk as a scalable alternative to GBIF.", bullet_style))

    # Build PDF
    doc.build(story)
    print(f"PDF report generated successfully at: {output_path}")

if __name__ == '__main__':
    out_dir = r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\docs"
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "hpai_gnn_project_progress_report_final.pdf")
    create_pdf_report(out_pdf)
