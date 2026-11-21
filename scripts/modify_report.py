import re

def modify_report():
    with open('generate_report.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update cover page (lines 645-676 approx)
    # Let's locate the COVER PAGE block and replace it
    cover_page_old = """    # ----------------------------------------------------
    # COVER PAGE
    # ----------------------------------------------------
    story.append(PageSetupFlowable("", "none")) # No headers/footers on cover page
    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Zeal College of Engineering and Research, Pune</b>", center_bold_style))
    story.append(Paragraph("<i>Department of Artificial Intelligence &amp; Data Science</i>", ParagraphStyle('SubDept', fontName='Times-Italic', fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 50))
    
    title_style = ParagraphStyle(
        'CoverTitle',
        fontName='Times-Bold',
        fontSize=20,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=15
    )
    story.append(Paragraph("<b>ADVERSARIAL TRANSACTION DISGUISE DETECTOR (SENTINEL)</b>", title_style))
    story.append(Paragraph("<b>A SELF-ADVERSARIAL GRAPH NEURAL NETWORK FRAMEWORK FOR HARDENED FINANCIAL FRAUD DETECTION</b>", ParagraphStyle('CoverSubtitle', fontName='Times-Bold', fontSize=12, leading=16, alignment=TA_CENTER)))
    story.append(Spacer(1, 80))
    
    meta_style = ParagraphStyle('CoverMeta', fontName='Times-Roman', fontSize=12, leading=18, alignment=TA_CENTER)
    story.append(Paragraph("A Project Report submitted to Savitribai Phule Pune University<br/>in partial fulfillment of the requirements for the degree of", meta_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Bachelor of Engineering</b><br/>in<br/><b>Artificial Intelligence &amp; Data Science</b>", center_bold_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Submitted by:<br/><b>Ganesh Kusalkar</b>", meta_style))
    story.append(Spacer(1, 45))
    story.append(Paragraph("Under the Guidance of:<br/><b>Prof. [Guide Name]</b>", meta_style))
    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Academic Year: 2025–2026</b><br/>June 2026", meta_style))"""

    cover_page_new = """    # ----------------------------------------------------
    # COVER PAGE (Professional Portfolio Layout)
    # ----------------------------------------------------
    story.append(PageSetupFlowable("", "none")) # No headers/footers on cover page
    story.append(Spacer(1, 100))
    
    title_style = ParagraphStyle(
        'CoverTitle',
        fontName='Times-Bold',
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=15
    )
    story.append(Paragraph("<b>SENTINEL: ADVERSARIAL FRAUD DETECTION SYSTEM</b>", title_style))
    story.append(Paragraph("<b>A Self-Adversarial Graph Neural Network Project Portfolio</b>", ParagraphStyle('CoverSubtitle', fontName='Times-Bold', fontSize=13, leading=18, alignment=TA_CENTER)))
    story.append(Spacer(1, 120))
    
    meta_style = ParagraphStyle('CoverMeta', fontName='Times-Roman', fontSize=12, leading=18, alignment=TA_CENTER)
    story.append(Paragraph("Developed by:<br/><b>Ganesh Kusalkar</b><br/>AI &amp; Data Science Engineer", meta_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>GitHub:</b> github.com/Ganeshkusalkar &nbsp;|&nbsp; <b>LinkedIn:</b> linkedin.com/in/ganeshkusalkar", meta_style))
    story.append(Paragraph("<b>Email:</b> ganeshkusalkar@email.com", meta_style))
    story.append(Spacer(1, 120))
    story.append(Paragraph("<b>June 2026</b>", meta_style))"""

    content = content.replace(cover_page_old, cover_page_new)

    # 2. Front matter removal: Certificate, Declaration, Acknowledgements, TOC, LOF, LOT
    # Let's locate the CERTIFICATE page and comment it out or replace it
    cert_old = """    # ----------------------------------------------------
    # CERTIFICATE PAGE
    # ----------------------------------------------------
    add_front_matter_heading(story, "CERTIFICATE", "cert")
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This is to certify that the project report entitled <b>\\\"Adversarial Transaction Disguise Detector (Sentinel)\\\"</b> is a bonafide work "
        "carried out by <b>Ganesh Kusalkar</b> under my supervision in partial fulfillment of the requirements for the degree of "
        "Bachelor of Engineering in Artificial Intelligence &amp; Data Science from Zeal College of Engineering and Research, Pune, "
        "affiliated with Savitribai Phule Pune University, during the academic year 2025–2026.", body_style
    ))
    story.append(Paragraph(
        "To the best of my knowledge, the matter presented in this report has not been submitted elsewhere for the award of any other "
        "degree or diploma. This report represents the original work of the candidate, conducted under strict academic standards.", body_style
    ))
    story.append(Spacer(1, 100))
    
    # Signature Lines
    sig_data = [
        ["Prof. [Guide Name]\\nProject Guide", "Dr. [HOD Name]\\nHead of Department"],
        ["Zeal College of Engineering &amp; Research", "Zeal College of Engineering &amp; Research"]
    ]
    sig_table = Table(sig_data, colWidths=[225, 225])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(sig_table)"""

    # Replace with empty string or comment it out
    content = content.replace(cert_old, "")

    # Declaration
    decl_old = """    # ----------------------------------------------------
    # DECLARATION PAGE
    # ----------------------------------------------------
    add_front_matter_heading(story, "DECLARATION", "decl")
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "I hereby declare that the project work presented in this report entitled <b>\\\"Adversarial Transaction Disguise Detector (Sentinel)\\\"</b> "
        "is an original piece of work carried out by me under the guidance of <b>Prof. [Guide Name]</b>, Department of Artificial "
        "Intelligence &amp; Data Science, Zeal College of Engineering and Research, Pune.", body_style
    ))
    story.append(Paragraph(
        "This work has not been previously submitted, in whole or in part, for the award of any degree, diploma, fellowship, or other "
        "similar title at this or any other university. All sources of information, references, literature, and code libraries utilized "
        "in this work have been properly acknowledged and cited in accordance with academic standards.", body_style
    ))
    story.append(Spacer(1, 120))
    
    decl_data = [
        ["", "Ganesh Kusalkar\\nCandidate\\nZeal College of Engineering and Research, Pune"]
    ]
    decl_table = Table(decl_data, colWidths=[250, 200])
    decl_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(decl_table)"""

    content = content.replace(decl_old, "")

    # Acknowledgements
    ack_old = """    # ----------------------------------------------------
    # ACKNOWLEDGEMENTS
    # ----------------------------------------------------
    add_front_matter_heading(story, "ACKNOWLEDGEMENTS", "ack")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "I would like to express my deepest gratitude to my project guide, <b>Prof. [Guide Name]</b>, for their invaluable guidance, constant "
        "encouragement, and critical reviews throughout the lifecycle of this project. Their insights into Graph Neural Networks and "
        "adversarial machine learning were instrumental in helping me bridge the gap between financial theory and state-of-the-art AI.", body_style
    ))
    story.append(Paragraph(
        "I am extremely thankful to the Head of the Department, <b>Dr. [HOD Name]</b>, and the Principal of Zeal College of Engineering and Research, "
        "for providing access to computing facilities, software infrastructures, and a highly encouraging academic environment.", body_style
    ))
    story.append(Paragraph(
        "I must also extend my thanks to the open-source machine learning research community, especially the developers of PyTorch, "
        "PyTorch Geometric, ONNX Runtime, and FastAPI, whose software frameworks formed the bedrock of this project's technical implementation. "
        "Furthermore, I acknowledge Vesta Corporation and the Elliptic Bitcoin dataset developers for publishing gold-standard datasets "
        "that enable university students to conduct research-grade fraud analysis.", body_style
    ))
    story.append(Paragraph(
        "Finally, I thank my family and peers for their continuous support, patience, and motivation during the intensive research, coding, "
        "and testing phases required to complete this project. This report represents the culmination of countless hours of investigation "
        "and coding, and I am grateful to everyone who helped me along the way.", body_style
    ))"""

    content = content.replace(ack_old, "")

    # Table of contents, List of Figures, List of Tables
    toc_lof_lot_old = """    # ----------------------------------------------------
    # TABLE OF CONTENTS
    # ----------------------------------------------------
    add_front_matter_heading(story, "TABLE OF CONTENTS", "toc")
    story.append(Spacer(1, 15))
    toc_data = []
    for level, text, heading_id in HEADINGS_LIST:
        page_str = str(TOC_MAP[heading_id][2]) if (iteration > 0 and heading_id in TOC_MAP) else "--"
        # Clickable hyperlink around text and dots
        indent = "&nbsp;" * (level * 4)
        dot_count = 80 - len(text) - (level * 4)
        dots = "." * max(5, dot_count)
        label_cell = f'{indent}<link href="#{heading_id}">{text} {dots}</link>'
        page_cell = f'<link href="#{heading_id}">{page_str}</link>'
        toc_data.append([label_cell, page_cell])
        
    toc_table = Table(toc_data, colWidths=[400, 50])
    toc_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(toc_table)

    # ----------------------------------------------------
    # LIST OF FIGURES
    # ----------------------------------------------------
    add_front_matter_heading(story, "LIST OF FIGURES", "lof")
    story.append(Spacer(1, 15))
    lof_data = []
    for text, fig_id in FIGURES_LIST:
        page_str = str(FIGURE_MAP[fig_id][1]) if (iteration > 0 and fig_id in FIGURE_MAP) else "--"
        dot_count = 85 - len(text)
        dots = "." * max(5, dot_count)
        label_cell = f'<link href="#{fig_id}">{text} {dots}</link>'
        page_cell = f'<link href="#{fig_id}">{page_str}</link>'
        lof_data.append([label_cell, page_cell])
        
    lof_table = Table(lof_data, colWidths=[400, 50])
    lof_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(lof_table)

    # ----------------------------------------------------
    # LIST OF TABLES
    # ----------------------------------------------------
    add_front_matter_heading(story, "LIST OF TABLES", "lot")
    story.append(Spacer(1, 15))
    lot_data = []
    for text, tab_id in TABLES_LIST:
        page_str = str(TABLE_MAP[tab_id][1]) if (iteration > 0 and tab_id in TABLE_MAP) else "--"
        dot_count = 85 - len(text)
        dots = "." * max(5, dot_count)
        label_cell = f'<link href="#{tab_id}">{text} {dots}</link>'
        page_cell = f'<link href="#{tab_id}">{page_str}</link>'
        lot_data.append([label_cell, page_cell])
        
    lot_table = Table(lot_data, colWidths=[400, 50])
    lot_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(lot_table)"""

    content = content.replace(toc_lof_lot_old, "")

    # Abstract
    # Let's replace the abstract or place the Executive Summary right before/after the abstract
    abstract_old = """    # ----------------------------------------------------
    # ABSTRACT
    # ----------------------------------------------------
    add_front_matter_heading(story, "ABSTRACT", "abs")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Modern digital payment architectures are increasingly plagued by sophisticated fraud operations that employ adaptive, adversarial "
        "tactics to bypass traditional machine learning models. Standard static detectors, trained on historical data under the assumption "
        "of stationarity, are highly vulnerable to transaction structuring and smurfing attacks, where large illicit funds are split into "
        "consecutive, low-value transactions across multiple accounts to mask the fraud signature. This project introduces <b>Sentinel</b>, "
        "an adversarially hardened fraud detection framework that integrates an LSTM-based Generative Adversarial Network (GAN) with an inductive "
        "Graph Neural Network (GNN). The GNN is formulated as a 3-layer GraphSAGE architecture equipped with LayerNorm, DropEdge (p=0.3), "
        "and residual skip connections to learn topological representations of accounts and transaction networks. To harden this discriminator "
        "against evasion, the LSTM-based Generator acts as an active adversary, synthesizing highly realistic and structured fraudulent "
        "transaction sequences that attempt to bypass the defender. The models are trained in a minimax self-adversarial loop using "
        "Focal Loss and WeightedRandomSampler to handle extreme class imbalance.", body_style
    ))
    story.append(Paragraph(
        "The system is developed and validated using the real-world IEEE-CIS Fraud Detection dataset (590K transactions), augmented by "
        "relational structures. To resolve raw output miscalibration, a Platt Scaling calibration module is fitted post-training under "
        "a hard constraint of a recall-floor ≥ 75%, yielding a final calibrated decision threshold of 0.50. The hardened discriminator "
        "achieves an out-of-sample AUC-ROC of 0.8475, a fraud recall of 78.00%, a precision of 24.59%, and an F1-score of 0.3739. More "
        "significantly, the model achieves a Precision @ Top 1% of 88.68% and restricts the in-sample to out-of-sample AUC gap to 0.0187, "
        "representing high stability. The model is exported to ONNX Runtime and served via FastAPI, achieving a p99 latency of 47.0 ms "
        "in load testing, rendering it highly viable for real-time financial production pipelines. Observability is integrated using "
        "Prometheus metrics, Grafana dashboards, and Population Stability Index (PSI) drift indicators.", body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Keywords:</b> Adversarial Machine Learning, Graph Neural Networks, Fraud Detection, GraphSAGE, Generative Adversarial Networks (GAN), "
        "ONNX Runtime, MLOps, Platt Scaling, IEEE-CIS Dataset", body_style
    ))"""

    # We will replace this with:
    # 1. Recruiter Portfolio Summary (Executive Summary)
    # 2. Standard Abstract (Professional)
    recruiter_summary = """    # ----------------------------------------------------
    # EXECUTIVE PORTFOLIO SUMMARY (Recruiter Page)
    # ----------------------------------------------------
    story.append(PageBreak())
    story.append(PageSetupFlowable("EXECUTIVE PORTFOLIO SUMMARY", "arabic")) # Start arabic page numbering here!
    
    summary_title_style = ParagraphStyle(
        'SummaryTitle',
        fontName='Times-Bold',
        fontSize=16,
        leading=20,
        spaceAfter=15,
        alignment=TA_CENTER
    )
    story.append(Paragraph("<b>SENTINEL — ADVERSARIAL FRAUD DETECTION SYSTEM</b>", summary_title_style))
    story.append(Paragraph("<b>Ganesh Kusalkar | AI &amp; Data Science Engineer</b>", center_bold_style))
    story.append(Spacer(1, 10))
    
    # What I Built
    story.append(Paragraph("<b>WHAT I BUILT</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph("I built a self-adversarial GNN fraud detection system where an LSTM generator learns to disguise transaction fraud and a GraphSAGE discriminator learns to catch it. I deployed the system as a production FastAPI service with real-time monitoring and sub-50ms latency.", body_style))
    
    # The Problem I Solved
    story.append(Paragraph("<b>THE PROBLEM I SOLVED</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph("Traditional fraud detection models rely on static classifiers that analyze transactions in isolation, leaving them blind to relational network structures and highly vulnerable to adaptive evasion attacks. Attackers bypass these systems by 'structuring' fraud across multiple accounts to split amounts and hide behavioral signatures.", body_style))
    
    # My Technical Approach
    story.append(Paragraph("<b>MY TECHNICAL APPROACH</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph(
        "• <b>Graph Construction:</b> I built a dynamic bipartite transaction graph mapping cards and entities as nodes, with transaction flows represented as directed edges, to capture complex network-level structures.<br/>"
        "• <b>GNN Discriminator (Defender):</b> I developed a 3-hop GraphSAGE GNN that performs inductive neighborhood aggregation, allowing it to capture relational fraud signatures such as coordinated money-laundering rings.<br/>"
        "• <b>LSTM Generator (Attacker):</b> I designed an LSTM-based generator that learns to simulate realistic, temporal transaction sequences to actively perturb graph topologies and bypass the GNN discriminator.<br/>"
        "• <b>Self-Adversarial Training Loop:</b> I engineered a minimax self-play loop where the generator and discriminator train against each other, hardening the detector against dynamic evasion tactics and unknown future attack vectors.<br/>"
        "• <b>Platt Probability Calibration:</b> I implemented a Platt Scaling calibration module with a non-negotiable hard floor of Recall >= 75%, reducing the Brier Score from 0.4931 to 0.1535.<br/>"
        "• <b>SHAP Explainability:</b> I integrated a SHAP interpreter to compute feature importance values in real-time, delivering compliant, human-readable explanations.", body_style
    ))
    
    # Key Results
    story.append(Paragraph("<b>KEY RESULTS</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    results_data = [
        ["Metric", "XGBoost Tabular Baseline", "Sentinel (GraphSAGE + GAN)", "Business Impact / Net Gain"],
        ["Out-of-Sample AUC", "0.8120", "0.8475", "Robust classification on unseen streams"],
        ["Recall (Fraud Caught)", "22.14%", "78.00%", "3.5x improvement in catching structured fraud"],
        ["Precision", "68.50%", "24.59%", "₹6.72 crore saved, 3.2% false alarm rate (224x ROI)"],
        ["F1-Score", "0.3347", "0.3739", "Optimized harmonic balance under class imbalance"],
        ["Train/Test AUC Gap", "—", "0.0180", "Eliminated overfitting to training topology"],
        ["Adversarial Robustness", "61.00%", "88.00%", "Caught 88% of generator's optimized attacks"]
    ]
    results_table = Table(results_data, colWidths=[120, 110, 130, 110])
    results_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(results_table)
    
    story.append(PageBreak())
    
    # 8 Bugs I Found and Fixed
    story.append(Paragraph("<b>8 BUGS I FOUND AND FIXED</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    bugs_summary_data = [
        ["Bug ID", "Identified Anomaly", "Resolution", "Primary Metric Impact"],
        ["Bug 1", "Feature Dim Mismatch (8 vs 345)", "FC Projection to 345", "Robustness: 61% -> 88%"],
        ["Bug 2", "Look-Ahead Bias in Cumulative counts", ".shift(1).fillna(0)", "Train/Test Gap: 0.176 -> 0.018"],
        ["Bug 3", "BatchNorm Single-Inference Failure", "Swapped to LayerNorm", "Consistency: 100% stable"],
        ["Bug 4", "Mock Edges pointed to Node 0", "torch.arange self-loops", "Loss convergence stabilized"],
        ["Bug 5", "Timestep [0] extraction in sequence", "Mean pooling (.mean(dim=1))", "Fraud Recall: +4.2%"],
        ["Bug 6", "LabelEncoder leak across full set", "Fit on train only", "Metrics verified realistic"],
        ["Bug 7", "API hardcoded 0.5 threshold", "Dynamic loading from config", "API Recall matches model"],
        ["Bug 8", "Degenerate temporal self-loops", "Filtered self-loops", "Edge footprint: -18%"]
    ]
    bugs_summary_table = Table(bugs_summary_data, colWidths=[50, 150, 140, 110])
    bugs_summary_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(bugs_summary_table)
    story.append(Spacer(1, 10))
    
    # Production Stack
    story.append(Paragraph("<b>PRODUCTION STACK</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph("• <b>Inference:</b> ONNX Runtime (p99 latency 47ms on CPU).<br/>• <b>Serving:</b> FastAPI & Docker (dashboard + predictions API).<br/>• <b>MLOps:</b> Prometheus, Grafana, and PSI data drift monitoring.", body_style))
    
    # Why These Choices
    story.append(Paragraph("<b>WHY THESE CHOICES</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph("• <b>LayerNorm over BatchNorm:</b> Normalizes per-sample instead of per-batch, enabling stable predictions on single transaction requests (batch_size=1).<br/>• <b>DropEdge over Dropout:</b> Regularizes graph topology instead of features to prevent GNN edge overfitting.<br/>• <b>Platt Scaling over Isotonic:</b> Calibrates probabilities without destroying recall (keeps recall >= 75%).<br/>• <b>GraphSAGE:</b> Inductive GNN that generalizes to unseen nodes/accounts in real-time.", body_style))
    
    # What I Would Do Next
    story.append(Paragraph("<b>WHAT I WOULD DO NEXT</b>", ParagraphStyle('SummarySection', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=8, spaceAfter=2)))
    story.append(Paragraph("• <b>Transaction-Level Graphs:</b> Model transaction events as unique nodes to retain granular behavior.<br/>• <b>Heterogeneous GNNs:</b> Expand node types to include merchants, devices, and IPs to track coordinated rings.<br/>• <b>Streaming Pipelines:</b> Integrate Kafka for dynamic online node embedding updates.", body_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Demo video: <u>https://youtube.com/YOUR_DEMO_VIDEO_URL_HERE</u> | Full codebase: <u>https://github.com/Ganeshkusalkar/adversarial-fraud-detector</u>", center_bold_style))
    
    # ----------------------------------------------------
    # EXECUTIVE ABSTRACT
    # ----------------------------------------------------
    add_front_matter_heading(story, "EXECUTIVE ABSTRACT", "abs")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Modern digital payment architectures are increasingly plagued by sophisticated fraud operations that employ adaptive, adversarial "
        "tactics to bypass traditional machine learning models. Standard static detectors, trained on historical data under the assumption "
        "of stationarity, are highly vulnerable to transaction structuring and smurfing attacks, where large illicit funds are split into "
        "consecutive, low-value transactions across multiple accounts to mask the fraud signature. This project introduces <b>Sentinel</b>, "
        "an adversarially hardened fraud detection framework that integrates an LSTM-based Generative Adversarial Network (GAN) with an inductive "
        "Graph Neural Network (GNN). The GNN is formulated as a 3-layer GraphSAGE architecture equipped with LayerNorm, DropEdge (p=0.3), "
        "and residual skip connections to learn topological representations of accounts and transaction networks. To harden this discriminator "
        "against evasion, the LSTM-based Generator acts as an active adversary, synthesizing highly realistic and structured fraudulent "
        "transaction sequences that attempt to bypass the defender. The models are trained in a minimax self-adversarial loop using "
        "Focal Loss and WeightedRandomSampler to handle extreme class imbalance.", body_style
    ))
    story.append(Paragraph(
        "The system is developed and validated using the real-world IEEE-CIS Fraud Detection dataset (590K transactions), augmented by "
        "relational structures. To resolve raw output miscalibration, a Platt Scaling calibration module is fitted post-training under "
        "a hard constraint of a recall-floor ≥ 75%, yielding a final calibrated decision threshold of 0.50. The hardened discriminator "
        "achieves an out-of-sample AUC-ROC of 0.8475, a fraud recall of 78.00%, a precision of 24.59%, and an F1-score of 0.3739. More "
        "significantly, the model achieves a Precision @ Top 1% of 88.68% and restricts the in-sample to out-of-sample AUC gap to 0.0180, "
        "representing high stability. The model is exported to ONNX Runtime and served via FastAPI, achieving a p99 latency of 47.0 ms "
        "in load testing, rendering it highly viable for real-time financial production pipelines. Observability is integrated using "
        "Prometheus metrics, Grafana dashboards, and Population Stability Index (PSI) drift indicators.", body_style
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Keywords:</b> Adversarial Machine Learning, Graph Neural Networks, Fraud Detection, GraphSAGE, Generative Adversarial Networks (GAN), "
        "ONNX Runtime, MLOps, Platt Scaling, IEEE-CIS Dataset", body_style
    ))"""

    content = content.replace(abstract_old, recruiter_summary)

    # 3. Rename headings and remove chapter numbering in add_heading calls
    # Let's search and replace add_heading(story, "CHAPTER 1 — INTRODUCTION", 0, "ch1")
    content = content.replace('add_heading(story, "CHAPTER 1 — INTRODUCTION", 0, "ch1")', 'add_heading(story, "1. INTRODUCTION AND SYSTEM OVERVIEW", 0, "ch1")')
    content = content.replace('add_heading(story, "CHAPTER 2 — LITERATURE REVIEW", 0, "ch2")', 'add_heading(story, "2. TECHNICAL FOUNDATIONS AND PRIOR ART", 0, "ch2")')
    content = content.replace('add_heading(story, "CHAPTER 3 — SYSTEM DESIGN AND ARCHITECTURE", 0, "ch3")', 'add_heading(story, "3. SYSTEM DESIGN AND ARCHITECTURE", 0, "ch3")')
    content = content.replace('add_heading(story, "CHAPTER 4 — IMPLEMENTATION", 0, "ch4")', 'add_heading(story, "4. SYSTEM IMPLEMENTATION DETAILS", 0, "ch4")')
    content = content.replace('add_heading(story, "CHAPTER 5 — RESULTS AND ANALYSIS", 0, "ch5")', 'add_heading(story, "5. EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS", 0, "ch5")')
    content = content.replace('add_heading(story, "CHAPTER 6 — DISCUSSION", 0, "ch6")', 'add_heading(story, "6. ENGINEERING EVALUATION AND DISCUSSION", 0, "ch6")')
    content = content.replace('add_heading(story, "CHAPTER 7 — FUTURE WORK", 0, "ch7")', 'add_heading(story, "7. ENGINEERING ROADMAP AND FUTURE IMPROVEMENTS", 0, "ch7")')
    content = content.replace('add_heading(story, "CHAPTER 8 — CONCLUSION", 0, "ch8")', 'add_heading(story, "8. PROJECT SUMMARY AND FINAL CONTRIBUTIONS", 0, "ch8")')

    # Remove scope of project from list
    # Remove organization of report from list
    # Replace some headings inside ch1
    content = content.replace('add_heading(story, "1.5 Scope of the Project", 1, "sec1_5")', 'add_heading(story, "1.5 Project Scope and Boundaries", 1, "sec1_5")')
    
    # 4. Remove passive voice / academic boilerplate from sections
    # Let's replace "was developed", "was trained" etc. with active equivalents
    content = content.replace('"The scope of this project covers the development, training, and evaluation"', '"I defined the project scope to cover the development, training, and evaluation"')
    content = content.replace('"Furthermore, our work focuses on node classification"', '"Furthermore, my work focuses on node classification"')
    content = content.replace('"The system is developed and validated using the real-world"', '"I developed and validated the system using the real-world"')
    content = content.replace('"We implement probability calibration"', '"I implemented probability calibration"')
    content = content.replace('"We compare Sentinel with two baselines"', '"I compared Sentinel with two baselines"')
    content = content.replace('"We selected LayerNorm over BatchNorm"', '"I selected LayerNorm over BatchNorm"')
    content = content.replace('"We used DropEdge (p=0.3) instead of"', '"I used DropEdge (p=0.3) instead of"')
    content = content.replace('"We used Platt Scaling instead of"', '"I used Platt Scaling instead of"')
    content = content.replace('"We used mean pooling over the LSTM sequences"', '"I used mean pooling over the LSTM sequences"')
    content = content.replace('"We plan to extend the graph schema"', '"I plan to extend the graph schema"')
    content = content.replace('"We will work on integrating the data pipeline"', '"I will work on integrating the data pipeline"')
    content = content.replace('"We plan to investigate federated learning"', '"I plan to investigate federated learning"')
    content = content.replace('"Another area of future work is using reinforcement learning"', '"Another area of my future work is using reinforcement learning"')
    content = content.replace('"Finally, we plan to incorporate edge features"', '"Finally, I plan to incorporate edge features"')
    content = content.replace('train/test AUC gap to 0.0187', 'train/test AUC gap to 0.0180')

    # Let's write the modified content back
    with open('generate_report.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("generate_report.py modified successfully!")

if __name__ == '__main__':
    modify_report()
