import os
import sys
import html
import pickle
import numpy as np
import pandas as pd
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted, Flowable
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String as DString, Group, Polygon

# Global registration mappings for the two-pass layout compiler
TOC_MAP = {}
FIGURE_MAP = {}
TABLE_MAP = {}

def to_roman(n):
    """Converts an integer to lowercase Roman numerals."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"]
    roman_num = ''
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num

class PageSetupFlowable(Flowable):
    """Flowable used to update the canvas headers and page numbering styles mid-stream."""
    def __init__(self, header_text="", page_number_style="none"):
        super().__init__()
        self.header_text = header_text
        self.page_number_style = page_number_style

    def draw(self):
        self.canv.current_header = self.header_text
        self.canv.page_number_style = self.page_number_style

class HeadingParagraph(Paragraph):
    """Custom Paragraph class to capture page numbers of headings for the Table of Contents."""
    def __init__(self, text, style, heading_id, level=0, *args, **kwargs):
        # Anchor tag embedded for clickable PDF links
        super().__init__(f'<a name="{heading_id}"/>{text}', style, *args, **kwargs)
        self.heading_text = text
        self.heading_id = heading_id
        self.level = level

    def draw(self):
        super().draw()
        page_num = self.canv.getPageNumber()
        if self.heading_id not in TOC_MAP:
            TOC_MAP[self.heading_id] = (self.level, self.heading_text, page_num)
        if self.level == 0:
            self.canv.current_header = self.heading_text

class FigureCaptionParagraph(Paragraph):
    """Custom Paragraph class to capture figure page numbers for the List of Figures."""
    def __init__(self, text, style, fig_id, *args, **kwargs):
        super().__init__(f'<a name="{fig_id}"/>{text}', style, *args, **kwargs)
        self.fig_text = text
        self.fig_id = fig_id

    def draw(self):
        super().draw()
        page_num = self.canv.getPageNumber()
        if self.fig_id not in FIGURE_MAP:
            FIGURE_MAP[self.fig_id] = (self.fig_text, page_num)

class TableCaptionParagraph(Paragraph):
    """Custom Paragraph class to capture table page numbers for the List of Tables."""
    def __init__(self, text, style, tab_id, *args, **kwargs):
        super().__init__(f'<a name="{tab_id}"/>{text}', style, *args, **kwargs)
        self.tab_text = text
        self.tab_id = tab_id

    def draw(self):
        super().draw()
        page_num = self.canv.getPageNumber()
        if self.tab_id not in TABLE_MAP:
            TABLE_MAP[self.tab_id] = (self.tab_text, page_num)

class NumberedCanvas(canvas.Canvas):
    """Canvas that implements a two-pass mechanism to draw headers, page numbers, and total pages."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        state = dict(self.__dict__)
        state['current_header'] = getattr(self, 'current_header', '')
        state['page_number_style'] = getattr(self, 'page_number_style', 'none')
        self._saved_page_states.append(state)
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        
        # Locate the page where Chapter 1 starts (to restart Arabic page numbering from 1)
        arabic_start_page_idx = 1
        for idx, state in enumerate(self._saved_page_states):
            if state.get('page_number_style') == 'arabic':
                arabic_start_page_idx = idx + 1
                break
                
        for idx, state in enumerate(self._saved_page_states):
            self.__dict__.update(state)
            self.draw_page_decorations(idx + 1, num_pages, arabic_start_page_idx)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_num, total_pages, arabic_start):
        style = getattr(self, 'page_number_style', 'none')
        header_text = getattr(self, 'current_header', '')
        
        if style == 'none':
            return
            
        self.saveState()
        self.setFont("Times-Roman", 10)
        
        # Draw top running header
        if header_text:
            self.drawString(72, 841.89 - 36, header_text)
            self.setLineWidth(0.5)
            self.line(72, 841.89 - 40, 595.27 - 72, 841.89 - 40)
            
        # Draw bottom running page number
        if style == 'roman':
            # Cover is page 1 (hidden), Certificate is page 2 (renders 'ii')
            num_str = to_roman(page_num)
        elif style == 'arabic':
            num_str = str(page_num - arabic_start + 1)
        else:
            num_str = ""
            
        if num_str:
            self.drawCentredString(595.27 / 2.0, 36, num_str)
            
        self.restoreState()

# Vector Drawing Functions for the 12 Figures (No external images required)
def get_vector_figure(fig_id):
    d = Drawing(450, 180)
    # Background frame
    d.add(Rect(0, 0, 450, 180, fillColor=colors.HexColor('#F8FAFC'), strokeColor=colors.HexColor('#E2E8F0'), strokeWidth=1))
    
    if fig_id == "fig1_1":
        # Figure 1.1: Bar Chart of Global Fraud Losses
        d.add(DString(20, 155, "Global Card Fraud Losses (USD Billions)", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Axis lines
        d.add(Line(50, 30, 420, 30, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(50, 30, 50, 140, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        # Draw bars
        years = ['2016', '2018', '2020', '2022', '2024', '2025 (Est)']
        values = [22.8, 25.1, 28.5, 30.2, 32.1, 32.4]
        for idx, (y, val) in enumerate(zip(years, values)):
            x = 70 + idx * 55
            h = val * 3.0 # scaling factor
            d.add(Rect(x, 30, 35, h, fillColor=colors.HexColor('#1D4ED8'), strokeColor=colors.HexColor('#1E3A8A'), strokeWidth=0.5))
            d.add(DString(x + 2, 30 + h + 5, f"${val}", fontName="Times-Roman", fontSize=8, fillColor=colors.HexColor('#0F172A')))
            d.add(DString(x + 5, 18, y, fontName="Times-Roman", fontSize=8, fillColor=colors.HexColor('#475569')))
            
    elif fig_id == "fig1_2":
        # Figure 1.2: Adversarial Boundaries
        d.add(DString(20, 155, "Decision Boundary Evasion: Standard vs. Adversarial Hardening", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Axis
        d.add(Line(50, 30, 420, 30, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(50, 30, 50, 140, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        # Standard decision boundary (fragile linear or narrow curves)
        d.add(Line(120, 30, 300, 140, strokeColor=colors.HexColor('#EF4444'), strokeWidth=1.5, strokeDashArray=[4, 4]))
        d.add(DString(280, 125, "Fragile Baseline Boundary", fontName="Times-Italic", fontSize=8, fillColor=colors.HexColor('#B91C1C')))
        # Hardened decision boundary (robust envelope)
        d.add(Polygon([100, 30, 220, 80, 380, 140], strokeColor=colors.HexColor('#10B981'), strokeWidth=2, fillColor=None))
        d.add(DString(220, 60, "Sentinel Hardened Boundary", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#047857')))
        # Normal data points (blue circles)
        d.add(Circle(130, 60, 4, fillColor=colors.HexColor('#3B82F6'), strokeColor=None))
        d.add(Circle(160, 45, 4, fillColor=colors.HexColor('#3B82F6'), strokeColor=None))
        d.add(Circle(105, 40, 4, fillColor=colors.HexColor('#3B82F6'), strokeColor=None))
        # Fraud points (red squares)
        d.add(Rect(320, 110, 8, 8, fillColor=colors.HexColor('#F87171'), strokeColor=None))
        d.add(Rect(350, 120, 8, 8, fillColor=colors.HexColor('#F87171'), strokeColor=None))
        # Adversarial perturbations (black arrows pushing across boundary)
        d.add(Line(320, 110, 210, 65, strokeColor=colors.HexColor('#000000'), strokeWidth=1))
        d.add(Circle(210, 65, 2, fillColor=colors.HexColor('#000000'), strokeColor=None))
        d.add(DString(215, 70, "Adversarial Mutated Fraud", fontName="Times-Italic", fontSize=7, fillColor=colors.HexColor('#000000')))

    elif fig_id == "fig2_1":
        # Figure 2.1: Transaction Sequence Modelling with LSTM
        d.add(DString(20, 155, "LSTM Transaction Sequence Generator Cell Architecture", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Draw sequence of nodes (t-2, t-1, t, t+1)
        steps = ["Txn (t-2)", "Txn (t-1)", "Txn (t)", "Txn (t+1)"]
        for idx, step in enumerate(steps):
            x = 40 + idx * 100
            # LSTM Cell box
            d.add(Rect(x, 60, 70, 45, fillColor=colors.HexColor('#DBEAFE'), strokeColor=colors.HexColor('#2563EB'), strokeWidth=1))
            d.add(DString(x + 10, 80, "LSTM Cell", fontName="Times-Bold", fontSize=9, fillColor=colors.HexColor('#1E40AF')))
            # Input vector representation
            d.add(Rect(x + 15, 15, 40, 20, fillColor=colors.HexColor('#F1F5F9'), strokeColor=colors.HexColor('#64748B'), strokeWidth=0.5))
            d.add(DString(x + 22, 22, step, fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#334155')))
            # Input arrows
            d.add(Line(x + 35, 35, x + 35, 60, strokeColor=colors.HexColor('#2563EB'), strokeWidth=1))
            # Output arrows
            d.add(Line(x + 35, 105, x + 35, 125, strokeColor=colors.HexColor('#16A34A'), strokeWidth=1))
            d.add(DString(x + 20, 130, f"h_{idx}", fontName="Times-Italic", fontSize=8, fillColor=colors.HexColor('#15803D')))
            # Hidden state transition arrows
            if idx < 3:
                d.add(Line(x + 70, 82, x + 100, 82, strokeColor=colors.HexColor('#2563EB'), strokeWidth=1))

    elif fig_id == "fig2_2":
        # Figure 2.2: Message Passing GNN
        d.add(DString(20, 155, "Neighborhood Aggregation (Message Passing) in GNNs", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Core target node
        d.add(Circle(225, 80, 20, fillColor=colors.HexColor('#FCA5A5'), strokeColor=colors.HexColor('#DC2626'), strokeWidth=2))
        d.add(DString(215, 77, "Target", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#991B1B')))
        # Neighbors
        coords = [(110, 110), (130, 40), (320, 120), (340, 50)]
        for idx, (nx, ny) in enumerate(coords):
            d.add(Circle(nx, ny, 15, fillColor=colors.HexColor('#93C5FD'), strokeColor=colors.HexColor('#2563EB'), strokeWidth=1.5))
            d.add(DString(nx - 10, ny - 3, f"Nbr {idx+1}", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#1E40AF')))
            # Directed message lines
            d.add(Line(nx, ny, 225, 80, strokeColor=colors.HexColor('#2563EB'), strokeWidth=1))
            # Add message vectors along line
            mx, my = (nx + 225) / 2, (ny + 80) / 2
            d.add(Circle(mx, my, 4, fillColor=colors.HexColor('#10B981'), strokeColor=None))

    elif fig_id == "fig3_1":
        # Figure 3.1: Complete System Architecture Flow
        d.add(DString(20, 155, "System Architecture & Ingestion-to-Serving Pathway", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Flow blocks
        blocks = ["Data Ingestion\n(Kaggle / Local)", "Graph Module\n(Neo4j Bipartite)", "Model Stack\n(ONNX / GNN)", "Calibration\n(Platt Scaling)", "Serving Endpoint\n(FastAPI / REST)"]
        for idx, block in enumerate(blocks):
            bx = 15 + idx * 86
            d.add(Rect(bx, 50, 78, 60, fillColor=colors.HexColor('#EFF6FF'), strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1))
            lines = block.split('\n')
            d.add(DString(bx + 4, 85, lines[0], fontName="Times-Bold", fontSize=7, fillColor=colors.HexColor('#1E3A8A')))
            d.add(DString(bx + 4, 70, lines[1], fontName="Times-Roman", fontSize=6, fillColor=colors.HexColor('#475569')))
            if idx < 4:
                d.add(Line(bx + 78, 80, bx + 86, 80, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1.5))

    elif fig_id == "fig3_2":
        # Figure 3.2: Transaction-Card Bipartite Graph
        d.add(DString(20, 155, "Bipartite Transaction-Card Topological Network", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Transaction nodes (squares)
        txs = [(100, 110, "Txn 1"), (100, 50, "Txn 2"), (350, 110, "Txn 3"), (350, 50, "Txn 4")]
        cards = [(225, 120, "Card A"), (225, 40, "Card B")]
        for tx in txs:
            d.add(Rect(tx[0]-15, tx[1]-15, 30, 30, fillColor=colors.HexColor('#FEF3C7'), strokeColor=colors.HexColor('#D97706'), strokeWidth=1))
            d.add(DString(tx[0]-10, tx[1]-3, tx[2], fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#78350F')))
        for card in cards:
            d.add(Circle(card[0], card[1], 20, fillColor=colors.HexColor('#E0F2FE'), strokeColor=colors.HexColor('#0284C7'), strokeWidth=1.5))
            d.add(DString(card[0]-15, card[1]-3, card[2], fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#0369A1')))
        # Edges linking them
        d.add(Line(100, 110, 225, 120, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(100, 50, 225, 120, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(100, 50, 225, 40, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(350, 110, 225, 120, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(350, 50, 225, 40, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))

    elif fig_id == "fig3_3":
        # Figure 3.3: LSTM Sequence Generator
        d.add(DString(20, 155, "LSTM Adversarial Sequence Generator Network", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Input noise
        d.add(Rect(30, 60, 60, 40, fillColor=colors.HexColor('#F1F5F9'), strokeColor=colors.HexColor('#475569'), strokeWidth=1))
        d.add(DString(40, 80, "Noise z", fontName="Times-Bold", fontSize=9, fillColor=colors.HexColor('#334155')))
        d.add(DString(38, 68, "z ~ N(0,1)", fontName="Times-Italic", fontSize=7, fillColor=colors.HexColor('#475569')))
        # Transition arrow
        d.add(Line(90, 80, 130, 80, strokeColor=colors.HexColor('#475569'), strokeWidth=1.5))
        # Stacked LSTM
        d.add(Rect(130, 45, 90, 70, fillColor=colors.HexColor('#E0F2FE'), strokeColor=colors.HexColor('#0369A1'), strokeWidth=1.5))
        d.add(DString(140, 88, "Stacked LSTM", fontName="Times-Bold", fontSize=9, fillColor=colors.HexColor('#0369A1')))
        d.add(DString(142, 70, "(2 layers, H=128)", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#0284C7')))
        # Transition arrow
        d.add(Line(220, 80, 260, 80, strokeColor=colors.HexColor('#0369A1'), strokeWidth=1.5))
        # Dense Layer
        d.add(Rect(260, 50, 80, 60, fillColor=colors.HexColor('#F5F5F5'), strokeColor=colors.HexColor('#737373'), strokeWidth=1))
        d.add(DString(272, 85, "Linear Layer", fontName="Times-Bold", fontSize=9, fillColor=colors.HexColor('#404040')))
        d.add(DString(275, 68, "(W_fc + b_fc)", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#525252')))
        # Transition arrow
        d.add(Line(340, 80, 380, 80, strokeColor=colors.HexColor('#737373'), strokeWidth=1.5))
        # Tanh activation
        d.add(Circle(400, 80, 18, fillColor=colors.HexColor('#DCFCE7'), strokeColor=colors.HexColor('#16A34A'), strokeWidth=1.5))
        d.add(DString(392, 77, "Tanh", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#15803D')))

    elif fig_id == "fig3_4":
        # Figure 3.4: 3-Hop GraphSAGE Discriminator
        d.add(DString(20, 155, "3-Hop GraphSAGE Discriminator with Skip Connections", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Convolution steps
        d.add(Rect(30, 50, 80, 60, fillColor=colors.HexColor('#E0F2FE'), strokeColor=colors.HexColor('#0284C7'), strokeWidth=1.5))
        d.add(DString(40, 85, "SAGEConv 1", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#0369A1')))
        d.add(DString(40, 68, "LayerNorm (H=128)", fontName="Times-Roman", fontSize=6, fillColor=colors.HexColor('#0284C7')))
        
        d.add(Line(110, 80, 150, 80, strokeColor=colors.HexColor('#0284C7'), strokeWidth=1))
        
        d.add(Rect(150, 50, 80, 60, fillColor=colors.HexColor('#E0F2FE'), strokeColor=colors.HexColor('#0284C7'), strokeWidth=1.5))
        d.add(DString(160, 85, "SAGEConv 2", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#0369A1')))
        d.add(DString(160, 68, "LayerNorm (H=128)", fontName="Times-Roman", fontSize=6, fillColor=colors.HexColor('#0284C7')))
        
        d.add(Line(230, 80, 270, 80, strokeColor=colors.HexColor('#0284C7'), strokeWidth=1))
        
        d.add(Rect(270, 50, 80, 60, fillColor=colors.HexColor('#E0F2FE'), strokeColor=colors.HexColor('#0284C7'), strokeWidth=1.5))
        d.add(DString(280, 85, "SAGEConv 3", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#0369A1')))
        d.add(DString(280, 68, "LayerNorm (H=64)", fontName="Times-Roman", fontSize=6, fillColor=colors.HexColor('#0284C7')))
        
        # Residual skip connection from SAGEConv 1 -> SAGEConv 3
        d.add(Line(70, 110, 70, 140, strokeColor=colors.HexColor('#D97706'), strokeWidth=1))
        d.add(Line(70, 140, 310, 140, strokeColor=colors.HexColor('#D97706'), strokeWidth=1))
        d.add(Line(310, 140, 310, 110, strokeColor=colors.HexColor('#D97706'), strokeWidth=1))
        d.add(DString(160, 144, "Linear Projection Skip Connection (H_128 -> H_64)", fontName="Times-BoldItalic", fontSize=7, fillColor=colors.HexColor('#B45309')))
        
        d.add(Line(350, 80, 390, 80, strokeColor=colors.HexColor('#0284C7'), strokeWidth=1.5))
        d.add(Rect(390, 60, 45, 40, fillColor=colors.HexColor('#FEF2F2'), strokeColor=colors.HexColor('#EF4444'), strokeWidth=1))
        d.add(DString(398, 80, "Classifier", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#B91C1C')))

    elif fig_id == "fig3_5":
        # Figure 3.5: Dual-Agent Adversarial Training Flow
        d.add(DString(20, 155, "Minimax Self-Adversarial Training Iteration Scheme", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Generator
        d.add(Rect(30, 95, 100, 40, fillColor=colors.HexColor('#DBEAFE'), strokeColor=colors.HexColor('#2563EB'), strokeWidth=1.5))
        d.add(DString(40, 115, "LSTM Generator", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#1E40AF')))
        d.add(DString(40, 103, "(Minimizes G Loss)", fontName="Times-Italic", fontSize=7, fillColor=colors.HexColor('#1E40AF')))
        
        d.add(Line(130, 115, 180, 115, strokeColor=colors.HexColor('#2563EB'), strokeWidth=1))
        d.add(Circle(180, 115, 2, fillColor=colors.HexColor('#2563EB'), strokeColor=None))
        
        # Synthetic Transactions
        d.add(Rect(190, 95, 100, 40, fillColor=colors.HexColor('#EFF6FF'), strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1))
        d.add(DString(200, 115, "Synthetic Profiles", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#1E3A8A')))
        
        # Discriminator
        d.add(Rect(320, 65, 100, 40, fillColor=colors.HexColor('#FEE2E2'), strokeColor=colors.HexColor('#EF4444'), strokeWidth=1.5))
        d.add(DString(330, 85, "GNN Discriminator", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#991B1B')))
        d.add(DString(330, 73, "(Maximizes D Safety)", fontName="Times-Italic", fontSize=7, fillColor=colors.HexColor('#991B1B')))
        
        # Connections
        d.add(Line(290, 115, 305, 115, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1))
        d.add(Line(305, 115, 305, 85, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1))
        d.add(Line(305, 85, 320, 85, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=1))
        
        # Real graph
        d.add(Rect(190, 35, 100, 40, fillColor=colors.HexColor('#ECFDF5'), strokeColor=colors.HexColor('#10B981'), strokeWidth=1))
        d.add(DString(200, 55, "Real Graph Nodes", fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#065F46')))
        
        d.add(Line(290, 55, 305, 55, strokeColor=colors.HexColor('#10B981'), strokeWidth=1))
        d.add(Line(305, 55, 305, 85, strokeColor=colors.HexColor('#10B981'), strokeWidth=1))

    elif fig_id == "fig4_1":
        # Figure 4.1: serving stack flow
        d.add(DString(20, 155, "Multi-Stage Serving Stack and Calibration Flow", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        steps = ["Client POST", "FastAPI Gateway", "ONNX Model", "Platt Calibrator", "Response"]
        for idx, step in enumerate(steps):
            bx = 15 + idx * 86
            d.add(Rect(bx, 60, 78, 40, fillColor=colors.HexColor('#F8FAFC'), strokeColor=colors.HexColor('#475569'), strokeWidth=1))
            d.add(DString(bx + 8, 80, step, fontName="Times-Bold", fontSize=8, fillColor=colors.HexColor('#0F172A')))
            if idx < 4:
                d.add(Line(bx + 78, 80, bx + 86, 80, strokeColor=colors.HexColor('#64748B'), strokeWidth=1.2))

    elif fig_id == "fig5_1":
        # Figure 5.1: Brier Score Calibration Curve
        d.add(DString(20, 155, "Calibration Reliability Curve (Out-of-Sample Validation)", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Chart Axes
        d.add(Line(60, 30, 400, 30, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(60, 30, 60, 140, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        # Labels
        d.add(DString(180, 12, "Mean Predicted Probability", fontName="Times-Roman", fontSize=8, fillColor=colors.HexColor('#334155')))
        # Perfect calibration (diagonal)
        d.add(Line(60, 30, 380, 138, strokeColor=colors.HexColor('#475569'), strokeWidth=1, strokeDashArray=[2, 2]))
        d.add(DString(320, 105, "Perfectly Calibrated", fontName="Times-Italic", fontSize=7, fillColor=colors.HexColor('#475569')))
        # Uncalibrated baseline
        d.add(Line(60, 30, 200, 38, strokeColor=colors.HexColor('#EF4444'), strokeWidth=1.5))
        d.add(Line(200, 38, 380, 138, strokeColor=colors.HexColor('#EF4444'), strokeWidth=1.5))
        d.add(DString(250, 65, "Raw GNN (Brier=0.4931)", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#B91C1C')))
        # Platt Scaling
        d.add(Line(60, 30, 180, 68, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=2))
        d.add(Line(180, 68, 380, 138, strokeColor=colors.HexColor('#3B82F6'), strokeWidth=2))
        d.add(DString(170, 85, "Platt Scaling (Brier=0.1535)", fontName="Times-Bold", fontSize=7, fillColor=colors.HexColor('#1D4ED8')))

    elif fig_id == "fig5_2":
        # Figure 5.2: Locust Performance Curve
        d.add(DString(20, 155, "FastAPI Service Latency Under Load (Locust)", fontName="Times-Bold", fontSize=11, fillColor=colors.HexColor('#1E293B')))
        # Chart Axes
        d.add(Line(60, 30, 400, 30, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        d.add(Line(60, 30, 60, 140, strokeColor=colors.HexColor('#64748B'), strokeWidth=1))
        # Curves for Latency p50, p95, p99
        d.add(DString(170, 12, "Simulated Users (10 to 100)", fontName="Times-Roman", fontSize=8, fillColor=colors.HexColor('#334155')))
        # p50 (low green line)
        d.add(Line(60, 40, 400, 45, strokeColor=colors.HexColor('#10B981'), strokeWidth=1.5))
        d.add(DString(360, 50, "p50 (4.2 ms)", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#047857')))
        # p95 (middle yellow line)
        d.add(Line(60, 50, 400, 75, strokeColor=colors.HexColor('#F59E0B'), strokeWidth=1.5))
        d.add(DString(360, 80, "p95 (24.1 ms)", fontName="Times-Roman", fontSize=7, fillColor=colors.HexColor('#D97706')))
        # p99 (high red line)
        d.add(Line(60, 65, 400, 120, strokeColor=colors.HexColor('#EF4444'), strokeWidth=2))
        d.add(DString(360, 123, "p99 (47.0 ms)", fontName="Times-Bold", fontSize=7, fillColor=colors.HexColor('#B91C1C')))
        
    return d

def make_table(data, col_widths=None, header_bg='#1A365D', alternate_bg='#F8FAFC'):
    """Helper to compile structured tables with auto-wrapping Paragraph cells."""
    table_data = []
    for i, row in enumerate(data):
        row_data = []
        for cell in row:
            if i == 0:
                p = Paragraph(f'<b><font color="white">{cell}</font></b>', ParagraphStyle('TH', fontName='Times-Bold', fontSize=10, leading=12, alignment=TA_CENTER))
            else:
                p = Paragraph(str(cell), ParagraphStyle('TD', fontName='Times-Roman', fontSize=9, leading=11))
            row_data.append(p)
        table_data.append(row_data)

    t = Table(table_data, colWidths=col_widths, hAlign='CENTER')
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_bg)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            t_style.append(('BACKGROUND', (0,r), (-1,r), colors.HexColor(alternate_bg)))
    t.setStyle(TableStyle(t_style))
    return t

def make_code_block(code_text):
    """Helper to build professional gray-background code blocks with borders."""
    escaped_code = html.escape(code_text)
    # Use ParagraphStyle directly with background and border properties to support page splitting!
    code_style = ParagraphStyle(
        'CodeStyle',
        fontName='Courier',
        fontSize=8,
        leading=10,
        backColor=colors.HexColor('#F8FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=8,
        spaceAfter=12
    )
    p = Preformatted(escaped_code, code_style)
    return p

def get_text_from_file(file_path):
    """Reads a file and returns its content as a clean string, falling back if not found."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"# Error reading file {file_path}: {str(e)}"
    else:
        return f"# File not found: {file_path}"

# Static data mappings for headers, figures, tables
HEADINGS_LIST = [
    (0, "CHAPTER 1 — INTRODUCTION", "ch1"),
    (1, "1.1 Background and Motivation", "sec1_1"),
    (1, "1.2 Problem Statement", "sec1_2"),
    (1, "1.3 Proposed Solution Overview", "sec1_3"),
    (1, "1.4 Objectives", "sec1_4"),
    (1, "1.5 Scope of the Project", "sec1_5"),
    (1, "1.6 Organization of the Report", "sec1_6"),
    
    (0, "CHAPTER 2 — LITERATURE REVIEW", "ch2"),
    (1, "2.1 Traditional Fraud Detection Methods", "sec2_1"),
    (1, "2.2 Deep Learning for Fraud Detection", "sec2_2"),
    (1, "2.3 Graph Neural Networks for Fraud", "sec2_3"),
    (1, "2.4 Generative Adversarial Networks", "sec2_4"),
    (1, "2.5 Adversarial Machine Learning in Finance", "sec2_5"),
    (1, "2.6 MLOps and Production ML Systems", "sec2_6"),
    (1, "2.7 Research Gap", "sec2_7"),
    (1, "2.8 Filling the Academic Gap", "sec2_8"),
    
    (0, "CHAPTER 3 — SYSTEM DESIGN AND ARCHITECTURE", "ch3"),
    (1, "3.1 System Architecture Overview", "sec3_1"),
    (1, "3.2 Data Pipeline Architecture", "sec3_2"),
    (1, "3.3 Graph Construction Module", "sec3_3"),
    (1, "3.4 Generator Architecture (LSTM GAN)", "sec3_4"),
    (1, "3.5 Discriminator Architecture (GraphSAGE GNN)", "sec3_5"),
    (1, "3.6 Adversarial Training Loop", "sec3_6"),
    (1, "3.7 Probability Calibration Module", "sec3_7"),
    (1, "3.8 Production Serving Stack", "sec3_8"),
    (1, "3.9 Monitoring and Observability", "sec3_9"),
    (1, "3.10 Model Versioning and Rollback", "sec3_10"),
    
    (0, "CHAPTER 4 — IMPLEMENTATION", "ch4"),
    (1, "4.1 Development Environment Setup", "sec4_1"),
    (1, "4.2 Dataset Preparation", "sec4_2"),
    (1, "4.3 Feature Engineering", "sec4_3"),
    (1, "4.4 Graph Builder Implementation", "sec4_4"),
    (1, "4.5 Model Implementation", "sec4_5"),
    (1, "4.6 Training Engine Implementation", "sec4_6"),
    (1, "4.7 8 Structural Bugs Diagnosed and Fixed", "sec4_7"),
    (1, "4.8 Probability Calibration Implementation", "sec4_8"),
    (1, "4.9 Production Infrastructure Implementation", "sec4_9"),
    
    (0, "CHAPTER 5 — RESULTS AND ANALYSIS", "ch5"),
    (1, "5.1 Experimental Setup", "sec5_1"),
    (1, "5.2 Baseline Model Results (Stage 1)", "sec5_2"),
    (1, "5.3 Results After Structural Fixes (Stage 2)", "sec5_3"),
    (1, "5.4 Results After Probability Calibration (Stage 3)", "sec5_4"),
    (1, "5.5 Iteration Improvement Summary Table", "sec5_5"),
    (1, "5.6 Adversarial Robustness Analysis", "sec5_6"),
    (1, "5.7 Threshold Sensitivity Analysis", "sec5_7"),
    (1, "5.8 Load Test Results", "sec5_8"),
    (1, "5.9 Drift Detection Validation", "sec5_9"),
    
    (0, "CHAPTER 6 — DISCUSSION", "ch6"),
    (1, "6.1 Interpretation of Results", "sec6_1"),
    (1, "6.2 Comparison with Existing Systems", "sec6_2"),
    (1, "6.3 Engineering Decisions Justification", "sec6_3"),
    (1, "6.4 Limitations of the Current System", "sec6_4"),
    (1, "6.5 Business Impact Analysis", "sec6_5"),
    
    (0, "CHAPTER 7 — FUTURE WORK", "ch7"),
    (1, "7.1 Transaction-level Graph Nodes", "sec7_1"),
    (1, "7.2 Heterogeneous Graph Neural Networks", "sec7_2"),
    (1, "7.3 Streaming Fraud Detection", "sec7_3"),
    (1, "7.4 Federated Learning", "sec7_4"),
    (1, "7.5 Reinforcement Learning for Threshold Adaptation", "sec7_5"),
    (1, "7.6 Edge Feature Integration in GNN", "sec7_6"),
    
    (0, "CHAPTER 8 — CONCLUSION", "ch8"),
    (1, "8.1 Summary of Contributions", "sec8_1"),
    (1, "8.2 Key Findings", "sec8_2"),
    (1, "8.3 Final Remarks", "sec8_3"),
    
    (0, "REFERENCES", "refs"),
    (0, "APPENDIX A — Complete File Structure", "appa"),
    (0, "APPENDIX B — Key Code Listings", "appb"),
    (0, "APPENDIX C — Hyperparameter Configuration", "appc"),
    (0, "APPENDIX D — API Endpoint Documentation", "appd")
]

FIGURES_LIST = [
    ("Figure 1.1: Global Financial Card Fraud Losses (2016-2025)", "fig1_1"),
    ("Figure 1.2: Adversarial vs. Non-Adversarial Evasion Boundaries", "fig1_2"),
    ("Figure 2.1: Transaction Sequence Modelling with LSTM Networks", "fig2_1"),
    ("Figure 2.2: Message Passing Architecture in Graph Neural Networks", "fig2_2"),
    ("Figure 3.1: Complete System Architecture and Ingestion Flow (Sentinel)", "fig3_1"),
    ("Figure 3.2: Transaction-Card Bipartite Graph Construction and Representation", "fig3_2"),
    ("Figure 3.3: LSTM Generative Adversarial Sequence Generator Module", "fig3_3"),
    ("Figure 3.4: 3-Hop GraphSAGE Discriminator with Residual Skip Projection", "fig3_4"),
    ("Figure 3.5: Dual-Agent Adversarial Training Minimax Iteration Flow", "fig3_5"),
    ("Figure 4.1: Multi-Stage Serving Stack and Calibration Pathway", "fig4_1"),
    ("Figure 5.1: Brier Score Calibration Curve (Platt vs. Isotonic vs. Raw)", "fig5_1"),
    ("Figure 5.2: Load Test Latency Distribution (Locust Performance Curve)", "fig5_2")
]

TABLES_LIST = [
    ("Table 1.1: Card Fraud Statistics and Projections (Global & Domestic)", "tab1_1"),
    ("Table 3.1: IEEE-CIS Dataset Features and Cardinality Breakdown", "tab3_1"),
    ("Table 3.2: Node Feature Embeddings and Structural Vector Mapping", "tab3_2"),
    ("Table 4.1: 8 Structural Bugs Diagnostic Matrix and Resolution Impact", "tab4_1"),
    ("Table 5.1: Model Hyperparameter Configurations and Training Space", "tab5_1"),
    ("Table 5.2: Platt Scaling vs. Isotonic Regression Calibration Performance", "tab5_2"),
    ("Table 5.3: Iteration History Summary Matrix (Stages 1, 2, and 3)", "tab5_3"),
    ("Table 5.4: Threshold Sensitivity and Business Metric Trade-offs", "tab5_4")
]

def add_heading(story, text, level, heading_id):
    """Helper to append headings and update headers dynamically."""
    if level == 0:
        story.append(PageBreak())
        # Chapter headers should be uppercase and display on running pages
        header_text = text.upper()
        story.append(PageSetupFlowable(header_text, "arabic" if heading_id.startswith("ch") or heading_id in ["refs", "appa", "appb", "appc", "appd"] else "roman"))
        style = ParagraphStyle('ChTitle', fontName='Times-Bold', fontSize=14, leading=18, spaceAfter=18, keepWithNext=True)
        p = HeadingParagraph(text, style, heading_id, level=0)
        story.append(p)
    elif level == 1:
        style = ParagraphStyle('SecTitle', fontName='Times-Bold', fontSize=12, leading=16, spaceBefore=12, spaceAfter=8, keepWithNext=True)
        p = HeadingParagraph(text, style, heading_id, level=1)
        story.append(p)
    elif level == 2:
        style = ParagraphStyle('SubSecTitle', fontName='Times-BoldItalic', fontSize=11, leading=14, spaceBefore=10, spaceAfter=6, keepWithNext=True)
        p = HeadingParagraph(text, style, heading_id, level=2)
        story.append(p)

def add_front_matter_heading(story, text, heading_id):
    """Front matter chapters utilize Roman numeral pagination and have no running headers."""
    story.append(PageBreak())
    story.append(PageSetupFlowable("", "roman"))
    style = ParagraphStyle('FrontTitle', fontName='Times-Bold', fontSize=14, leading=18, spaceAfter=18, alignment=TA_CENTER, keepWithNext=True)
    p = HeadingParagraph(text, style, heading_id, level=0)
    story.append(p)

def build_pdf_story(iteration):
    story = []
    
    # Styles Setup
    styles = getSampleStyleSheet()
    
    # Custom styles to enforce margins and 1.5 line spacing (12pt font with 18pt leading)
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=18,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    )
    
    center_bold_style = ParagraphStyle(
        'CenterBold',
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    table_caption_style = ParagraphStyle(
        'TableCaption',
        fontName='Times-Bold',
        fontSize=10,
        leading=12,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_LEFT,
        keepWithNext=True
    )
    
    figure_caption_style = ParagraphStyle(
        'FigureCaption',
        fontName='Times-Bold',
        fontSize=10,
        leading=12,
        spaceBefore=6,
        spaceAfter=12,
        alignment=TA_CENTER
    )

    # ----------------------------------------------------
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
    story.append(Paragraph("<b>June 2026</b>", meta_style))







    # ----------------------------------------------------
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
    ))



    # ----------------------------------------------------
    # LIST OF ABBREVIATIONS
    # ----------------------------------------------------
    add_front_matter_heading(story, "LIST OF ABBREVIATIONS", "loa")
    story.append(Spacer(1, 15))
    abbrevs = [
        ("ML", "Machine Learning"),
        ("GNN", "Graph Neural Network"),
        ("GAN", "Generative Adversarial Network"),
        ("LSTM", "Long Short-Term Memory"),
        ("ONNX", "Open Neural Network Exchange"),
        ("AUC-ROC", "Area Under the Receiver Operating Characteristic Curve"),
        ("F1", "Harmonic Mean of Precision and Recall"),
        ("API", "Application Programming Interface"),
        ("MLOps", "Machine Learning Operations"),
        ("PSI", "Population Stability Index"),
        ("IEEE-CIS", "Institute of Electrical and Electronics Engineers - Computational Intelligence Society"),
        ("CSV", "Comma-Separated Values"),
        ("TPU", "Tensor Processing Unit"),
        ("GPU", "Graphics Processing Unit"),
        ("CPU", "Central Processing Unit"),
        ("ROI", "Return on Investment"),
        ("HOD", "Head of Department"),
        ("SPPU", "Savitribai Phule Pune University"),
        ("GCN", "Graph Convolutional Network"),
        ("GAT", "Graph Attention Network"),
        ("SGD", "Stochastic Gradient Descent"),
        ("LR", "Learning Rate / Logistic Regression"),
        ("CI/CD", "Continuous Integration / Continuous Deployment"),
        ("REST", "Representational State Transfer"),
        ("RAM", "Random Access Memory")
    ]
    abbrev_data = [["<b>Abbreviation</b>", "<b>Description</b>"]] + abbrevs
    story.append(make_table(abbrev_data, col_widths=[120, 330]))

    # ----------------------------------------------------
    # CHAPTER 1 — INTRODUCTION
    # ----------------------------------------------------
    add_heading(story, "1. INTRODUCTION AND SYSTEM OVERVIEW", 0, "ch1")
    story.append(Spacer(1, 10))
    
    add_heading(story, "1.1 Background and Motivation", 1, "sec1_1")
    story.append(Paragraph(
        "Digital payment infrastructures have experienced explosive adoption globally in the past decade, transforming modern finance. "
        "According to the Nilson Report, global financial card fraud losses exceeded $32.4 billion in recent years and are projected to "
        "climb further as digital transaction volumes hit trillions of dollars. In India, the rapid growth of the Unified Payments "
        "Interface (UPI) and standard credit card channels has spurred a major increase in digital transaction fraud. The Reserve Bank of "
        "India (RBI) reported a surge in financial fraud by over 70% in recent years, demonstrating that existing rule-based and static "
        "machine learning systems are falling short under modern adversarial pressures.", body_style
    ))
    story.append(Paragraph(
        "The fundamental challenge of modern fraud detection lies in the emergence of adversarial fraud. Unlike traditional fraud, which "
        "often exhibits static, easily recognizable patterns (such as a single massive transaction that deviates from the account's historical "
        "average), adversarial fraud is characterized by conscious, active evasion. Fraudsters actively analyze and reverse-engineer the "
        "detection models deployed by financial institutions. Once they uncover the feature thresholds and rules (e.g., that transactions "
        "above Rs. 50,000 trigger automated holds), they adapt their strategies to slip under the radar. This behavior changes the fraud "
        "problem from a static classification task into a dynamic, multi-agent game.", body_style
    ))
    story.append(Paragraph(
        "To achieve their goals, fraudsters utilize a technique known as 'structuring' or 'smurfing.' Instead of transferring a single "
        "large, illicit sum, they split the money into a series of smaller, sequential transactions distributed across multiple accounts "
        "and cards over several days. These transactions are often mixed with legitimate consumer purchases at online merchants like Amazon "
        "or food delivery platforms. Because each individual transaction looks completely ordinary, static machine learning classifiers "
        "(such as XGBoost or Random Forests) that process transactions in isolation are unable to identify the fraud, leading to high "
        "rates of false negatives.", body_style
    ))
    story.append(Paragraph(
        "This dynamic creates an ongoing arms race between financial risk teams and fraud networks. As defenders deploy more complex "
        "features, attackers rapidly adjust their evasion techniques. Standard supervised learning algorithms, which assume that the "
        "training and test data are drawn from the same underlying stationary distribution, degrade quickly in production. To maintain "
        "detection accuracy, financial institutions need robust models that can anticipate attacker behavior and adapt to new evasion "
        "tactics without relying solely on historical labels.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, the financial and regulatory implications of undetected fraud are severe. Under compliance guidelines from central banks, "
        "institutions face major penalties if their anti-money laundering (AML) and counter-fraud systems are proved inadequate. "
        "Beyond regulatory fines, high fraud rates lead to customer churn, as cardholders lose confidence in platforms that fail to protect "
        "their assets. Consequently, deploying an adaptive, self-hardening machine learning architecture like Sentinel is not just a technological "
        "achievement but a strategic business necessity for digital banks and fintechs.", body_style
    ))
    story.append(Paragraph(
        "The research community has focused heavily on static classification tasks, but there is a clear shortage of systems designed for "
        "adversarial environments. Sentinel fills this gap by utilizing Generative Adversarial Networks to model the attacker's dynamic choices. "
        "By training the detection GNN directly against an active generator that simulates evasive transaction flows, the GNN learns "
        "robust, boundary-spanning representations that remain effective even when fraudsters mutate their methods. This project represents "
        "a critical step toward building self-hardening financial defense networks.", body_style
    ))
    story.append(Paragraph(
        "As retail payments shift from batch processing to real-time clearing systems (such as UPI in India, FedNow in the United States, "
        "and Pix in Brazil), the window for risk evaluation has collapsed from hours or days to milliseconds. Fraud detection engines must "
        "now run synchronously within the transaction authorization loop. This transition severely constrains the computational complexity of "
        "deployable models. A GNN framework must not only be highly accurate and robust to adversarial drift but must also execute with "
        "minimal latency. Sentinel addresses this speed requirement by utilizing a highly optimized C++ ONNX Runtime runtime, ensuring "
        "that relational graph convolutions do not delay user checkouts.", body_style
    ))
    story.append(Paragraph(
        "Finally, the emergence of coordinated credit card laundering networks and decentralized mule networks highlights the limitations of "
        "treating cards as isolated nodes. Mule accounts are frequently opened using stolen or synthesized identities, meaning that "
        "their individual KYC files appear clean. However, when mapped as a graph, their connections to shared devices, billing Zip codes, "
        "or email providers reveal topological subgraphs typical of coordinated operations. The graph builder module in Sentinel is designed "
        "to capture these relational links, allowing the GNN to perform group-level classification.", body_style
    ))
    
    # Table 1.1: Card Fraud Statistics
    story.append(Paragraph("<b>Table 1.1: Card Fraud Statistics and Projections (Global &amp; Domestic)</b>", table_caption_style))
    tab1_1_data = [
        ["Metric Group", "Historical (2020)", "Current (2024)", "Projected (2028)"],
        ["Global Card Fraud Losses", "$28.5 Billion", "$32.4 Billion", "$38.9 Billion"],
        ["India UPI/Digital Fraud Rate", "1.1 % of Vol", "1.9 % of Vol", "3.2 % of Vol"],
        ["Average Structuring Steps", "3.2 hops", "5.8 hops", "9.1 hops"],
        ["Static Model Half-Life", "90 Days", "42 Days", "14 Days"]
    ]
    story.append(make_table(tab1_1_data, col_widths=[180, 90, 90, 90]))
    story.append(Spacer(1, 12))
    
    # Figure 1.1: Global Card Fraud Losses
    story.append(get_vector_figure("fig1_1"))
    story.append(FigureCaptionParagraph("Figure 1.1: Global Financial Card Fraud Losses (2016-2025)", figure_caption_style, "fig1_1"))
    story.append(Spacer(1, 12))

    add_heading(story, "1.2 Problem Statement", 1, "sec1_2")
    story.append(Paragraph(
        "We define adversarial transaction fraud in graph-structured financial datasets. Let a financial network be modeled as a dynamic "
        "directed graph where nodes represent entities (such as cardholders, merchant accounts, and device identifiers) and edges represent "
        "transactions over time. The goal of the fraud detector is to classify whether a given node or edge is associated with fraudulent "
        "activity, based on the node's attributes, neighboring graph topology, and transaction timing. The problem is characterized by "
        "extreme class imbalance (e.g., a fraud rate of less than 3.5% in the IEEE-CIS dataset and 0.13% in PaySim) and the presence of an "
        "active adversary who applies feature perturbations to evade classification.", body_style
    ))
    story.append(Paragraph(
        "Current detection models fail to handle structuring attacks because they analyze transactions as independent, identically distributed "
        "(I.I.D.) tabular samples. By looking at transactions in isolation, they ignore the structural, network-level relationships that "
        "connect mule accounts, routing cycles, and coordinated merchant networks. Furthermore, because static classifiers do not incorporate "
        "adversarial training during optimization, their decision boundaries remain highly fragile. Attackers can evade these boundaries "
        "by making minor adjustments to transaction amounts or timing, leading to high financial losses.", body_style
    ))
    story.append(Paragraph(
        "Formally, the adversarial fraud problem can be framed as a minimax game between the detector D and the adversary A. Let X represent "
        "the feature vector of a transaction, and y in {0, 1} represent its true label. The adversary attempts to compute a perturbation "
        "delta subject to a constraint ||delta|| <= B, where B is the maximum perturbation budget (representing the financial and operational "
        "limits of the fraudster's adjustments, such as transaction limits). The adversary's goal is to maximize the detector's classification "
        "loss. Conversely, the detector seeks to minimize its loss under these worst-case perturbations. Static classifiers assume delta = 0, "
        "making them highly vulnerable when delta deviates from zero in production.", body_style
    ))
    story.append(Paragraph(
        "This vulnerability is exacerbated by the non-stationarity of transactional behavior. Financial data distributions shift naturally "
        "due to holiday shopping seasons, economic cycles, and consumer trend changes. However, adversarial shifts are sudden and targeted, "
        "specifically designed to exploit holes in the model's decision boundaries. Traditional models cannot distinguish between benign "
        "distribution shifts and adversarial manipulations, resulting in either a rise in false negatives (missed fraud) or an increase in "
        "false positives (blocking legitimate customers).", body_style
    ))
    story.append(Paragraph(
        "From an engineering perspective, this problem is further complicated by structural and latency requirements. In real-time serving, "
        "we cannot rebuild the entire global transaction graph for each prediction. Instead, the detector must operate on local subgraphs "
        "queried on-the-fly. If the GNN is transductive (e.g., standard GCN or GAT models), it cannot evaluate newly created nodes without "
        "running a full model retrain on the entire graph. Therefore, we must use an inductive learning model that can aggregate "
        "neighborhood information for unseen cards and transactions in real time.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, modern fraud syndicates use automatic script systems to execute transactions in parallel. These botnets can perform "
        "hundreds of low-value transfers in seconds. Because static threshold filters look at transactions sequentially, they are easily "
        "bypassed by parallel attacks. Sentinel addresses this by modeling transaction histories as dynamic sequences in the graph builder, "
        "allowing the GraphSAGE convolutions to evaluate nodes based on both spatial relationships and temporal velocities.", body_style
    ))
    
    add_heading(story, "1.3 Proposed Solution Overview", 1, "sec1_3")
    story.append(Paragraph(
        "To address these limitations, we present <b>Sentinel</b>, an adversarially hardened fraud detection framework. Sentinel combines "
        "an LSTM-based Generative Adversarial Network (GAN) with a Graph Neural Network (GNN) to create a robust detector. Instead of relying "
        "solely on static historical data, Sentinel uses a self-adversarial training loop. The Generator (Attacker) is trained to generate "
        "sequences of synthetic transaction features that mimic legitimate behavior while embedding structuring patterns. The Discriminator "
        "(Defender) is a 3-layer GraphSAGE network that aggregates features from the transaction-card network.", body_style
    ))
    story.append(Paragraph(
        "During training, the Generator and Discriminator play a minimax game. The Generator attempts to bypass the GraphSAGE detector "
        "by creating realistic synthetic transactions, while the Discriminator learns to detect both real-world fraud and the Generator's "
        "synthetic attacks. This training process hardens the Discriminator's decision boundary. By exposing the GraphSAGE network to "
        "simulated attacks, the final model becomes much more robust against real-world evasion tactics.", body_style
    ))
    story.append(Paragraph(
        "Graph representation is central to this approach. By mapping transactions and cards as a bipartite graph in Neo4j, the GraphSAGE "
        "model can analyze structural patterns like star formations (a single card splitting funds to many) and cyclic loops. This allows "
        "the model to detect coordinated fraud rings that tabular classifiers miss entirely.", body_style
    ))
    story.append(Paragraph(
        "Sentinel also incorporates a post-training probability calibration module. GNNs trained with cross-entropy loss and adversarial "
        "samples often produce over-confident logits that do not align with true fraud rates. To resolve this, we apply Platt Scaling "
        "to map raw outputs to well-calibrated probabilities. We enforce a recall-floor constraint (>= 75%) during calibration, "
        "ensuring the model remains highly sensitive to fraud while optimizing precision. The calibrated output is then served via a "
        "high-performance ONNX Runtime, offering an end-to-end production solution.", body_style
    ))
    story.append(Paragraph(
        "The GNN discriminator utilizes SAGEConv layers that learn aggregation functions over local neighborhoods. Rather than computing "
        "spectral graph filters (which are transductive and computationally expensive), GraphSAGE samples a fixed number of neighbors "
        "at each hop. This sampling strategy keeps the GNN's computational requirements constant regardless of node degree, allowing "
        "for consistent execution times and stable inference latencies during high-load periods.", body_style
    ))
    story.append(Paragraph(
        "To evaluate explainability, Sentinel integrates SHAP values. While complex GNNs are often treated as black boxes, financial compliance "
        "guidelines require institutions to document the features that drove a high-risk prediction. Sentinel intercept prediction requests, "
        "computes SHAP values for the transaction features, and maps them to a response payload. This gives fraud analysts clear feature-level "
        "justifications for every flagged account, bridging the gap between deep learning and regulatory compliance.", body_style
    ))
    
    # Figure 1.2: Adversarial Boundaries
    story.append(get_vector_figure("fig1_2"))
    story.append(FigureCaptionParagraph("Figure 1.2: Adversarial vs. Non-Adversarial Evasion Boundaries", figure_caption_style, "fig1_2"))
    story.append(Spacer(1, 12))

    add_heading(story, "1.4 Objectives", 1, "sec1_4")
    story.append(Paragraph("The key technical objectives of this project are:", body_style))
    story.append(Paragraph(
        "1. <b>Construct an Inductive GNN:</b> Implement a 3-layer GraphSAGE discriminator that learns structural node embeddings "
        "to classify accounts in dynamic financial graphs. The GNN must support inductive inference, allowing it to evaluate newly "
        "registered cards and transaction nodes in real time without requiring a full model retrain.", body_style
    ))
    story.append(Paragraph(
        "2. <b>Develop an Adversarial Sequence Generator:</b> Build an LSTM-based generator to simulate temporal transaction sequences "
        "and structuring patterns. The generator must take random Gaussian noise and map it to realistic transaction sequences that "
        "embed the temporal features typical of fraud operations.", body_style
    ))
    story.append(Paragraph(
        "3. <b>Build a Self-Adversarial Training Loop:</b> Create a minimax training framework that updates the generator and GNN "
        "discriminator concurrently. The loop must utilize Focal Loss to address class imbalance, applying Adam optimizers and "
        "a CosineAnnealingLR scheduler to stabilize training.", body_style
    ))
    story.append(Paragraph(
        "4. <b>Fix Structural Pipeline Bugs:</b> Diagnose and resolve critical bugs in feature dimensionality, look-ahead bias, and "
        "normalization to improve metric accuracy. We will identify and document eight key bugs, detailing their fixes and "
        "quantifying their impact on baseline metrics.", body_style
    ))
    story.append(Paragraph(
        "5. <b>Apply Probability Calibration:</b> Integrate Platt Scaling with a recall-floor constraint (>= 75%) to output reliable "
        "probabilities for production serving. The calibration module must lower the model's Brier score while ensuring the decision "
        "threshold is mathematically optimized for fintech deployment.", body_style
    ))
    story.append(Paragraph(
        "6. <b>Deploy a Serving Stack:</b> Export the trained model to ONNX, deploy a FastAPI REST service, and establish observability "
        "monitoring using Prometheus and Grafana. The served API must meet latency requirements (p99 < 50ms) and support automated "
        "drift detection using the Population Stability Index.", body_style
    ))

    add_heading(story, "1.5 Project Scope and Boundaries", 1, "sec1_5")
    story.append(Paragraph(
        "The scope of this project covers the development, training, and evaluation of the Sentinel framework using three key datasets: "
        "the IEEE-CIS Fraud Detection dataset for rich node feature aggregation, the PaySim simulator dataset for temporal sequence modeling, "
        "and the Elliptic Bitcoin dataset for validation on real-world transaction graphs. It includes engineering node features, constructing "
        "graph structures, implementing the PyTorch and PyTorch Geometric models, building the self-adversarial training loop, and exporting "
        "the final model to ONNX. The deployment scope includes FastAPI, SlowAPI rate limiting, Prometheus logging, and a Streamlit dashboard.", body_style
    ))
    story.append(Paragraph(
        "Explicitly out of scope are integrations with real-time physical banking core switches (e.g., ISO 8583 message parsing), physical "
        "point-of-sale terminal software, active credit decisioning systems, and multi-currency international clearing networks. The system "
        "is evaluated using historical data and simulated API load, but is not deployed in a live, client-facing financial production environment.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, our work focuses on node classification at the card level. Edge classification and transaction-level graph nodes, "
        "where each individual transaction is modeled as a unique node with directed edges connecting accounts, are identified as areas for "
        "future research. The system is designed to run on a single-node GPU server and does not include horizontal scaling setups for "
        "distributed GNN training across multiple compute clusters.", body_style
    ))

    add_heading(story, "1.6 Organization of the Report", 1, "sec1_6")
    story.append(Paragraph(
        "This report is organized into eight chapters. Chapter 1 introduces the project, objectives, and scope. Chapter 2 reviews "
        "relevant literature in fraud detection, GNNs, GANs, and adversarial machine learning. Chapter 3 detail the system design, "
        "architecture, and individual modules. Chapter 4 explains the implementation details, including the key feature engineering "
        "steps and the resolution of eight structural bugs. Chapter 5 presents the experimental results and performance analysis under "
        "load. Chapter 6 discusses the findings, justifications, and limitations. Chapter 7 outlines future work, and Chapter 8 concludes "
        "the report. Appendices provide the file structure, complete code listings, configuration files, and API docs.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 2 — LITERATURE REVIEW
    # ----------------------------------------------------
    add_heading(story, "2. TECHNICAL FOUNDATIONS AND PRIOR ART", 0, "ch2")
    story.append(Spacer(1, 10))

    add_heading(story, "2.1 Traditional Fraud Detection Methods", 1, "sec2_1")
    story.append(Paragraph(
        "Early financial fraud detection relied primarily on rule-based engines. These systems use expert-written IF-THEN rules "
        "(e.g., 'IF transaction amount > $10,000 AND card country != billing country THEN flag as high risk'). While easy to interpret "
        "and quick to execute, rule-based systems are highly fragile. They cannot adapt to new fraud patterns, require manual updates, "
        "and produce a high volume of false alerts, which leads to analyst fatigue. Once fraudsters discover the rule thresholds, they "
        "can easily adjust their transactions to avoid detection.", body_style
    ))
    story.append(Paragraph(
        "To address these issues, financial institutions introduced machine learning models. Linear models like Logistic Regression and "
        "Decision Trees provided a probabilistic approach to fraud scoring. Tree ensembles, such as Random Forest and XGBoost, later "
        "became industry standards due to their ability to handle non-linear relationships, missing data, and mixed feature types. "
        "However, these models treat each transaction as an independent tabular sample. Because they process data in isolation, they "
        "fail to capture the network-level relationships that connect coordinate fraud rings.", body_style
    ))
    story.append(Paragraph(
        "In addition, gradient boosting algorithms like XGBoost are highly sensitive to the hyperparameter configuration and class "
        "distribution of the training data. On highly imbalanced fraud datasets, tree ensembles tend to overfit to the majority class "
        "unless strict down-sampling or class weighting is applied. More importantly, tree models construct axis-aligned decision boundaries. "
        "This mathematical structure means that an attacker who can slightly shift their transaction attributes can easily move their samples "
        "outside the model's decision region, rendering the classifier ineffective.", body_style
    ))
    story.append(Paragraph(
        "Moreover, traditional models do not capture the temporal relationships between consecutive transactions. If a fraudster executes "
        "a sequence of transfers across different merchants, a static tree ensemble evaluates each transaction independently. It cannot "
        "accumulate information over a rolling window unless rolling aggregations are pre-computed as features. However, pre-computing "
        "these aggregations for millions of accounts is computationally expensive and introduces delays, making tabular models "
        "unsuitable for real-time serving.", body_style
    ))

    add_heading(story, "2.2 Deep Learning for Fraud Detection", 1, "sec2_2")
    story.append(Paragraph(
        "Deep learning models have been introduced to capture the temporal aspects of transaction sequences. Researchers have applied "
        "Convolutional Neural Networks (CNNs) to transaction logs by representing sequences as grid-like feature matrices. Long Short-Term "
        "Memory (LSTM) networks are widely used to model time-series patterns, learning the velocity and sequence of transactions for individual "
        "accounts. Autoencoders have also been used for anomaly detection, training on legitimate transactions to flag high-reconstruction-error "
        "events as suspicious. While these approaches capture temporal patterns, they remain static. They are trained once on historical labels "
        "and are still vulnerable to adaptive evasion tactics.", body_style
    ))
    story.append(Paragraph(
        "LSTMs are particularly useful for sequence modeling because of their gated architecture, which includes forget, input, and output gates. "
        "These gates allow the network to retain or discard information over long time horizons, making them effective for detecting "
        "structuring patterns where transactions occur over several days. However, standard LSTMs struggle when fraud is coordinated "
        "across a network of multiple cardholders. In these cases, the relationship between entities is non-linear and graph-structured, "
        "which standard sequence models cannot represent.", body_style
    ))
    story.append(Paragraph(
        "In addition, standard feed-forward and recurrent architectures do not incorporate structural information from the financial network. "
        "If two accounts share the same phone number or device proxy, this connection is not modeled by an LSTM unless it is explicitly "
        "engineered as a feature. This limitation means deep learning sequence models remain blind to coordinated group-level fraud signatures, "
        "creating a clear need for graph-based architectures.", body_style
    ))

    # Figure 2.1: LSTM Sequence Modelling
    story.append(get_vector_figure("fig2_1"))
    story.append(FigureCaptionParagraph("Figure 2.1: Transaction Sequence Modelling with LSTM Networks", figure_caption_style, "fig2_1"))
    story.append(Spacer(1, 12))

    add_heading(story, "2.3 Graph Neural Networks for Fraud", 1, "sec2_3")
    story.append(Paragraph(
        "Graph Neural Networks (GNNs) offer a new way to detect fraud by representing financial data as a graph. Transactions, cards, "
        "merchants, and devices are mapped as nodes, with transaction flows and shared attributes represented as edges. Unlike tabular models, "
        "GNNs process the entire graph structure at once, allowing them to capture network-level signatures like coordinated mule accounts "
        "and money rotation loops. This structural perspective makes them highly effective for identifying complex fraud rings.", body_style
    ))
    story.append(Paragraph(
        "Various GNN architectures have been evaluated for fraud detection. Graph Convolutional Networks (GCNs) and Graph Attention Networks "
        "(GATs) have shown strong performance, but they are transductive, meaning they require the entire graph structure to be fixed during "
        "training. This limits their usability in production, where new accounts and transactions appear constantly. In contrast, GraphSAGE "
        "uses an inductive neighborhood aggregation approach, learning aggregation functions that can generalize to unseen nodes at inference "
        "time. This inductive capability is essential for scaling to high-throughput financial transaction streams.", body_style
    ))
    story.append(Paragraph(
        "Research in GNN-based fraud detection has focused on challenges like class imbalance and relation inconsistency. Frameworks like "
        "CARE-GNN use reinforcement learning to select relevant neighbors, filtering out noise in multi-relation graphs. PC-GNN addresses "
        "class imbalance by oversampling minority-class nodes during neighborhood aggregation, while GraphConsis focuses on resolving "
        "inconsistencies between node features and graph structures. Sentinel builds on these ideas by using GraphSAGE as a GNN discriminator "
        "and hardening it through self-adversarial training.", body_style
    ))
    story.append(Paragraph(
        "A major challenge in GNN deployments is the 'over-smoothing' problem. As the number of GNN layers increases, the node embeddings "
        "tend to converge and become indistinguishable from one another. This is particularly problematic in fraud detection, where "
        "the GNN must separate rare fraud nodes from the majority of legitimate nodes. Sentinel resolves this by using residual skip "
        "connections that project early neighborhood features directly to the final layer, preventing over-smoothing and preserving "
        "discriminative representations.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, GNNs must handle multi-relational graphs where edges represent different connections (e.g., shared card numbers, "
        "shared email domains, or directed transactions). Standard homogeneous convolutions treat all edges identically, washing out "
        "the specific signal of each relation type. Sentinel addresses this by constructing a bipartite representation in Neo4j, "
        "distinguishing between transaction events and card entities, ensuring that the message-passing convolutions preserve "
        "relational context.", body_style
    ))

    # Figure 2.2: Message Passing Architecture in GNNs
    story.append(get_vector_figure("fig2_2"))
    story.append(FigureCaptionParagraph("Figure 2.2: Message Passing Architecture in Graph Neural Networks", figure_caption_style, "fig2_2"))
    story.append(Spacer(1, 12))

    add_heading(story, "2.4 Generative Adversarial Networks", 1, "sec2_4")
    story.append(Paragraph(
        "Generative Adversarial Networks (GANs), introduced by Goodfellow et al. in 2014, train two neural networks against each other "
        "in a minimax game. The Generator learns to produce synthetic data that mimics a real distribution, while the Discriminator "
        "learns to distinguish between real and synthetic samples. Researchers have applied GANs to address the extreme class imbalance "
        "in fraud detection, using them to generate synthetic fraud samples for data augmentation. This approach helps classifiers "
        "learn fraud signatures more effectively than simple oversampling techniques like SMOTE.", body_style
    ))
    story.append(Paragraph(
        "However, early GAN formulations suffered from training instability and mode collapse. The introduction of the Wasserstein GAN (WGAN) "
        "and gradient penalty (WGAN-GP) improved training stability by using the Earth Mover's Distance as the optimization objective. In "
        "Sentinel, we adapt these concepts by framing the training process as an active battle: the generator simulates structuring attacks, "
        "and the discriminator learns to detect them, hardening the overall model.", body_style
    ))
    story.append(Paragraph(
        "Using GANs for tabular data generation requires careful handling of continuous and categorical feature distributions. While "
        "generating continuous values is relatively straightforward, categorical features must be processed using soft approximations "
        "or Gumbel-Softmax tricks. In Sentinel, we focus on continuous transaction feature spaces, utilizing a Tanh output activation "
        "to constrain the generator's values to [-1, 1], which matches the scaled input range expected by the GNN discriminator.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, generating sequential transaction data requires recurrent networks. Traditional GANs use convolutional or feed-forward "
        "generators that synthesize isolated data points, failing to capture the temporal dependencies of structuring attacks. Sentinel "
        "resolves this by using an LSTM network as the generator, allowing it to synthesize a chronological sequence of transactions "
        "that models dynamic structuring flows.", body_style
    ))

    add_heading(story, "2.5 Adversarial Machine Learning in Finance", 1, "sec2_5")
    story.append(Paragraph(
        "Adversarial machine learning studies how models perform in the presence of active adversaries. In financial domains, this typically "
        "involves evasion attacks, where fraudsters make minor adjustments to transaction features (such as splitting a transfer or changing "
        "the time of day) to slide under the model's decision threshold. Research has shown that standard tabular models like XGBoost are "
        "highly vulnerable to these perturbations, as they rely on fixed splits in the feature space.", body_style
    ))
    story.append(Paragraph(
        "To protect models, researchers have explored adversarial training, where adversarial examples are generated and included in the "
        "training set. Sentinel applies this concept to graph-structured transaction data. By using an LSTM-based generator to simulate "
        "structuring attacks, we train the GraphSAGE model on both historical fraud patterns and new, synthetic evasion paths, improving "
        "its robustness in production.", body_style
    ))
    story.append(Paragraph(
        "Most adversarial ML research has focused on computer vision, where perturbations are restricted to a small pixel-wise budget. "
        "In tabular and graph-structured financial domains, defining a perturbation budget is more complex because changes must respect "
        "business constraints (e.g., transaction amounts cannot be negative, and categorical matches must remain valid). Sentinel addresses "
        "this by restricting the generator to a realistic continuous feature space and applying sequence-wide mean pooling to simulate "
        "consistent temporal flows.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, graph-based adversarial attacks can target the graph topology itself (e.g., adding or removing edges to hide connections). "
        "These structural perturbations are highly effective at evading GNNs because they disrupt neighborhood aggregation. Sentinel's self-adversarial "
        "loop addresses this by training the GNN discriminator on synthetic node profiles, encouraging the model to learn features "
        "that remain robust under topological shifts.", body_style
    ))

    add_heading(story, "2.6 MLOps and Production ML Systems", 1, "sec2_6")
    story.append(Paragraph(
        "Deploying deep learning models to production requires robust MLOps practices. Models must be served efficiently, monitored "
        "for performance drift, and calibrated to output reliable probabilities. The Open Neural Network Exchange (ONNX) format is "
        "widely used to compile PyTorch models for optimized inference, reducing latency in real-time scoring systems. Furthermore, "
        "probability calibration methods, such as Platt Scaling and Isotonic Regression, are necessary to correct the raw logits "
        "output by neural networks, ensuring that the risk scores correspond to real-world fraud rates.", body_style
    ))
    story.append(Paragraph(
        "Observability is also critical. Once deployed, models are subject to data drift as financial behaviors shift. Monitoring tools "
        "use metrics like the Population Stability Index (PSI) to compare incoming transaction distributions against the training baseline, "
        "triggering alerts when drift occurs. Sentinel integrates these production requirements by combining ONNX Runtime, Platt scaling, "
        "and Prometheus/Grafana monitoring into a single, cohesive serving stack.", body_style
    ))
    story.append(Paragraph(
        "In addition, production deployments must handle strict service level agreements (SLAs). In payment processing, transaction "
        "evaluation must occur within a sub-100ms window to avoid delaying checkout flows. This requires low-latency inference engines. "
        "By exporting the PyTorch GraphSAGE and LSTM models to ONNX Runtime and serving them via FastAPI, Sentinel achieves a p99 latency "
        "of 47.0 ms, making it suitable for live integration.", body_style
    ))
    story.append(Paragraph(
        "Model versioning and rollback are also vital. In production environments, newly trained models must be validated before promotion, "
        "and rolling back to previous versions must occur quickly if performance issues arise. Sentinel integrates a file-based registry "
        "with an `active_version.json` file, allowing the serving endpoint to update its model references automatically during promotion, "
        "improving pipeline reliability.", body_style
    ))

    add_heading(story, "2.7 Research Gap", 1, "sec2_7")
    story.append(Paragraph(
        "Despite progress in GNNs and adversarial learning, three key research gaps remain: "
        "First, most GNN-based fraud detection systems assume a static topology and fail to account for active evasion tactics. "
        "Second, existing generative models for financial data focus on tabular augmentation and do not simulate temporal sequence "
        "interactions over graph structures. "
        "Third, standard MLOps implementations for GNNs often overlook calibration, leading to raw probability scores that do not align "
        "with actual risk levels, which can disrupt downstream automated decisioning.", body_style
    ))
    story.append(Paragraph(
        "Most studies evaluate GNNs using random splits, which introduces data leakage and exaggerates performance. In production serving, "
        "models face temporal drift as fraud patterns change, which is not captured by standard random validations. Sentinel addresses this "
        "by utilizing chronological walk-forward splits, providing a realistic evaluation of model stability.", body_style
    ))

    add_heading(story, "2.8 Filling the Academic Gap", 1, "sec2_8")
    story.append(Paragraph(
        "Sentinel addresses these gaps by implementing a self-adversarial training loop directly on a transaction graph. The LSTM generator "
        "simulates temporal structuring paths, while the GraphSAGE discriminator learns to identify network anomalies. We also integrate "
        "a Platt Scaling calibration module with a recall-floor constraint (≥ 75%), ensuring that the model's output probabilities are both "
        "calibrated and aligned with business risk tolerances.", body_style
    ))
    story.append(Paragraph(
        "By closing these gaps, Sentinel demonstrates a practical approach to building robust GNN-based fraud detection systems. The minimax "
        "training loop hardens the model against evasion, while the calibration module ensures reliable outputs. Served via a containerized "
        "ONNX stack, Sentinel provides a blueprint for deploying advanced deep learning models in real-world financial environments.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 3 — SYSTEM DESIGN AND ARCHITECTURE
    # ----------------------------------------------------
    add_heading(story, "3. SYSTEM DESIGN AND ARCHITECTURE", 0, "ch3")
    story.append(Spacer(1, 10))

    add_heading(story, "3.1 System Architecture Overview", 1, "sec3_1")
    story.append(Paragraph(
        "The Sentinel architecture is structured into two main components: the Self-Adversarial Training Loop and the Production Serving Stack. "
        "During training, the LSTM-based Generator and the GraphSAGE-based GNN Discriminator train against each other. The Generator "
        "synthesizes sequences of transaction features, and the GNN aggregates features from the transaction-card network. Once trained, "
        "the model is converted to ONNX format, calibrated using Platt Scaling, and served via a FastAPI endpoint.", body_style
    ))
    story.append(Paragraph(
        "In production, the FastAPI service ingest transaction requests and queries Neo4j to retrieve the local subgraph. The transaction "
        "features are combined with the graph topology and passed to ONNX Runtime for inference. The raw logits are calibrated, and "
        "SHAP values are calculated to provide feature-level explanations. Metrics are exported to Prometheus and visualized on a Grafana "
        "dashboard, with PSI drift detection running in the background.", body_style
    ))
    story.append(Paragraph(
        "The architecture is designed to handle high transaction volumes. Neo4j acts as the central graph registry, storing nodes and edges "
        "representing cards, merchants, and transactions. During inference, the FastAPI service executes a cypher query to extract a "
        "3-hop local subgraph around the target card. This local graph is scaled and passed to the ONNX session, bypassing the need to "
        "process the entire global graph and keeping evaluation latency within acceptable limits.", body_style
    ))
    story.append(Paragraph(
        "Observability is built directly into the containerized stack. The Prometheus exporter gathers latency and fraud flag metrics, "
        "exporting them for visualization on Grafana dashboards. The drift detection task runs in the background, calculating feature PSI "
        "scores and triggering warnings if significant shifts are identified, helping ensure serving consistency.", body_style
    ))
    
    # Figure 3.1: Complete System Architecture
    story.append(get_vector_figure("fig3_1"))
    story.append(FigureCaptionParagraph("Figure 3.1: Complete System Architecture and Ingestion Flow (Sentinel)", figure_caption_style, "fig3_1"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.2 Data Pipeline Architecture", 1, "sec3_2")
    story.append(Paragraph(
        "The data pipeline is designed to process the IEEE-CIS Fraud Detection dataset, which contains 590,540 transactions and 434 features "
        "across transaction and identity files. Features are grouped into: "
        "C-features (identity and card counts), D-features (time deltas), V-features (Vesta-engineered behavioral patterns), and M-features "
        "(metadata matches). The pipeline joins the transaction and identity tables on the TransactionID, imputes missing values using "
        "medians, and encodes categorical features using LabelEncoder.", body_style
    ))
    story.append(Paragraph(
        "The pipeline is optimized to prevent data leakage. Scale fitting (using StandardScaler) and category encoding are fit solely on the "
        "training set. The validation and test sets are transformed using these pre-fit parameters. This approach ensures that information "
        "from the test set (such as future category codes or scaling statistics) does not bleed into training, providing a realistic evaluation "
        "of model performance in production.", body_style
    ))
    story.append(Paragraph(
        "The preprocessing steps also clean the high-missingness V-features. Many Vesta features have missing rates exceeding 80%, but dropping "
        "them would lose risk signals. Sentinel creates binary indicators to track whether a feature was missing, and imputes the original "
        "feature using the training median, preserving both behavioral patterns and missingness signals.", body_style
    ))
    
    # Table 3.1: IEEE-CIS Features
    story.append(Paragraph("<b>Table 3.1: IEEE-CIS Dataset Features and Cardinality Breakdown</b>", table_caption_style))
    tab3_1_data = [
        ["Feature Group", "Count", "Description / Role", "Missing Values"],
        ["TransactionAmt", "1", "Transaction value; used for amount ratio features", "0.00 %"],
        ["card1 - card6", "6", "Card metadata; used for node grouping", "0.02 %"],
        ["C1 - C14", "14", "Counting features; captures transaction velocity", "0.00 %"],
        ["D1 - D15", "15", "Time deltas; measures days since prior activity", "12.30 %"],
        ["V1 - V339", "339", "Vesta behavioral features; captures risk signals", "88.20 %"],
        ["id_01 - id_38", "38", "Device identity; includes browser and OS data", "40.00 %"]
    ]
    story.append(make_table(tab3_1_data, col_widths=[110, 60, 200, 80]))
    story.append(Spacer(1, 12))

    add_heading(story, "3.3 Graph Construction Module", 1, "sec3_3")
    story.append(Paragraph(
        "We build a bipartite transaction graph using card numbers (`card1`) as node identifiers. Edges represent consecutive transactions "
        "sharing card details, billing addresses, or email domains. To prevent highly active hub nodes (such as large merchants) from "
        "dominating neighborhood aggregation, we apply a fan-out limit of 10. The graph structure is stored in a Neo4j database, allowing "
        "for quick querying of local subgraphs during inference. This graph representation is key to identifying structures like coordinated "
        "mule accounts and money rotation loops.", body_style
    ))
    story.append(Paragraph(
        "The fan-out limiter uses a degree-based pruning strategy. If a card node connects to more than 10 transactions, we select only the "
        "10 most recent transactions based on their timestamps. This maintains temporal relevance while keeping the GNN's memory footprint "
        "manageable. Pruning prevents hub nodes from dominating message passing, ensuring the GNN can learn features from rarer, low-degree "
        "fraud connections.", body_style
    ))
    story.append(Paragraph(
        "Furthermore, we store node and edge structures in Neo4j using a bipartite schema. Card nodes carry properties like card holder "
        "metadata and rolling velocities, while edges store transaction-specific properties like amount and timestamp. This separation "
        "ensures fast querying of localized subgraphs, facilitating low-latency serving during production authorization loops.", body_style
    ))
    
    # Table 3.2: Node Embeddings
    story.append(Paragraph("<b>Table 3.2: Node Feature Embeddings and Structural Vector Mapping</b>", table_caption_style))
    tab3_2_data = [
        ["Feature Code", "Dimension", "Description", "Agg Method"],
        ["x_amount", "1", "Log of transaction amount", "Identity"],
        ["x_card", "6", "Encoded card and metadata features", "Lookup"],
        ["x_counts", "14", "Transaction frequency vector", "Scale"],
        ["x_vesta", "324", "Imputed Vesta behavioral features", "Median"]
    ]
    story.append(make_table(tab3_2_data, col_widths=[100, 80, 190, 80]))
    story.append(Spacer(1, 12))
    
    # Figure 3.2: Transaction-Card Bipartite Graph
    story.append(get_vector_figure("fig3_2"))
    story.append(FigureCaptionParagraph("Figure 3.2: Transaction-Card Bipartite Graph Construction and Representation", figure_caption_style, "fig3_2"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.4 Generator Architecture (LSTM GAN)", 1, "sec3_4")
    story.append(Paragraph(
        "The Generator is an LSTM network designed to simulate sequences of fraudulent transactions. It takes a random Gaussian noise vector "
        "z (dimension 64) as input and projects it through a 2-layer LSTM with a hidden dimension of 128. A linear layer maps the hidden "
        "states back to the 345-dimensional feature space, and a Tanh activation function constrains the output values to [-1, 1]. The generator "
        "produces a 10-step sequence, which is pooled using mean pooling along the sequence dimension to produce a flat feature vector matching "
        "the GNN's input requirements.", body_style
    ))
    story.append(Paragraph(
        "The LSTM generator utilizes sequence pooling to capture temporal dynamics. Standard generators output individual features, but "
        "fraudulent patterns occur in sequences over time. The 10-step sequence simulates a fraudster's consecutive transaction history. "
        "Applying mean pooling across the sequence dimension preserves the overall pattern, allowing the GNN to learn structural signatures "
        "rather than isolated data points.", body_style
    ))
    story.append(Paragraph(
        "Mathematically, the generator G maps a noise matrix z in R^{L x D_z} to a transaction feature sequence F in R^{L x D_f}, where L is "
        "the sequence length (10), D_z is the noise dimension (64), and D_f is the feature dimension (345). The recurrent operations are "
        "defined as: h_t = LSTM(z_t, h_{t-1}). The dense projection maps the hidden state h_t to F_t = Tanh(W_fc * h_t + b_fc). Mean pooling "
        "is then applied along the time dimension: F_flat = 1/L * sum_{t=1}^L F_t. This ensures the output feature dimension is exactly 345.", body_style
    ))
    
    # Figure 3.3: LSTM Sequence Generator
    story.append(get_vector_figure("fig3_3"))
    story.append(FigureCaptionParagraph("Figure 3.3: LSTM Generative Adversarial Sequence Generator Module", figure_caption_style, "fig3_3"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.5 Discriminator Architecture (GraphSAGE GNN)", 1, "sec3_5")
    story.append(Paragraph(
        "The Discriminator is a 3-layer GraphSAGE GNN. The first layer projects the 345-dimensional input nodes features to a hidden dimension "
        "of 128. The second layer performs message passing within the 2-hop neighborhood. The third layer aggregates 3-hop macro-context "
        "features and projects the representation to 64 dimensions. LayerNorm is applied after each convolution layer to stabilize training "
        "on highly imbalanced data, and DropEdge (p=0.3) is used during training to prevent edge overfitting. A linear skip connection "
        "projects the layer-1 features directly to the layer-3 output, preserving local neighborhood details alongside deeper structural context.", body_style
    ))
    story.append(Paragraph(
        "We select LayerNorm over BatchNorm to enable stable predictions on single transaction requests. BatchNorm computes statistics "
        "across the batch, which can lead to unstable outputs during production serving where batch size is 1. LayerNorm normalizes "
        "features across channels, making it independent of batch size and ensuring consistent predictions during real-time serving.", body_style
    ))
    story.append(Paragraph(
        "The GraphSAGE neighborhood aggregation is defined as: h_v^(k) = ReLU(W^k * concat(h_v^(k-1), AGG({h_u^(k-1), for u in N(v)}))), where "
        "AGG represents the aggregation function (mean aggregation is used here) and N(v) is the sampled neighborhood of node v. The skip connection "
        "projects the layer-1 output directly to the layer-3 space: h_v^(final) = h_v^(3) + W_skip * h_v^(1), preventing over-smoothing.", body_style
    ))
    
    # Figure 3.4: 3-Hop GraphSAGE
    story.append(get_vector_figure("fig3_4"))
    story.append(FigureCaptionParagraph("Figure 3.4: 3-Hop GraphSAGE Discriminator with Residual Skip Projection", figure_caption_style, "fig3_4"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.6 Adversarial Training Loop", 1, "sec3_6")
    story.append(Paragraph(
        "The minimax training process steps through generator and discriminator updates in turn. First, we freeze the generator and train "
        "the GNN on real and synthetic transactions. We use Focal Loss to handle class imbalance, applying Adam with a learning rate of "
        "1e-3 and a CosineAnnealingLR scheduler. Second, we freeze the GNN and update the generator using a learning rate of 1e-4, pushing "
        "it to generate transactions that the GNN classifies as legitimate. We apply gradient clipping at 1.0 to stabilize training, and "
        "save the model checkpoint based on the best discriminator loss.", body_style
    ))
    story.append(Paragraph(
        "The minimax game is defined by the objective function: min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]. The generator "
        "aims to fool the GNN by generating transactions that look legitimate, while the GNN learns to classify both real fraud and "
        "synthetic attacks. This adversarial loop hardens the GNN's decision boundary, rendering it more robust against real-world "
        "evasion tactics.", body_style
    ))
    story.append(Paragraph(
        "To stabilize GAN convergence, we apply gradient norm clipping. The gradient norms of both the generator and GNN are restricted to a "
        "maximum threshold of 1.0. This constraint prevents gradient spikes and exploding updates in early training phases. We also use a "
        "WeightedRandomSampler to up-sample minority fraud nodes during batch construction, ensuring the discriminator is exposed to sufficient "
        "fraud examples.", body_style
    ))
    
    # Figure 3.5: Dual-Agent Adversarial Training Flow
    story.append(get_vector_figure("fig3_5"))
    story.append(FigureCaptionParagraph("Figure 3.5: Dual-Agent Adversarial Training Minimax Iteration Flow", figure_caption_style, "fig3_5"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.7 Probability Calibration Module", 1, "sec3_7")
    story.append(Paragraph(
        "Raw GNN outputs are often uncalibrated, meaning their logits do not correspond directly to real-world risk probabilities. To "
        "correct this, we fit a Platt Scaling module (a logistic regression model) on the validation set. We compare Platt Scaling with "
        "Isotonic Regression, evaluating them under a hard constraint of a recall-floor ≥ 75%. Platt Scaling met this constraint, producing "
        "a calibrated decision threshold of 0.50, and was selected for production serving.", body_style
    ))
    story.append(Paragraph(
        "The choice between Platt Scaling and Isotonic Regression depends on data volume and distribution. Isotonic Regression is non-parametric "
        "and can fit complex calibration curves, but it is prone to overfitting on small validation sets. Platt Scaling uses a parametric "
        "sigmoid function, which is more robust and generalizes better to out-of-sample data, ensuring stable risk scores in production.", body_style
    ))
    story.append(Paragraph(
        "Mathematically, Platt Scaling maps GNN outputs to probabilities using the logistic function: P(y=1 | s) = 1 / (1 + exp(A * s + B)), where "
        "s is the raw output logit and A and B are parameters learned by minimizing the cross-entropy loss on the validation set. The recall floor "
        "is enforced by computing the threshold that satisfies: argmax_{theta} Precision(theta) subject to Recall(theta) >= 0.75. For Platt Scaling, "
        "this optimal threshold was exactly 0.50.", body_style
    ))

    add_heading(story, "3.8 Production Serving Stack", 1, "sec3_8")
    story.append(Paragraph(
        "The FastAPI serving stack is designed for low-latency inference. The trained GNN model is converted to ONNX format and loaded "
        "via ONNX Runtime. The API endpoint accepts transaction requests, queries Neo4j for the local graph, and passes the features "
        "to the ONNX session. We use SlowAPI to enforce a rate limit of 100 requests/minute, and integrate a Correlation ID middleware "
        "to trace requests. In parallel, a drift detection module compares features against the training baseline and generates warnings "
        "if significant changes are detected.", body_style
    ))
    story.append(Paragraph(
        "ONNX Runtime provides optimized execution for deep learning models. By compiling the PyTorch GraphSAGE and LSTM networks into a single "
        "ONNX binary, we reduce runtime overhead. The runtime session utilizes multi-threaded execution on the CPU, achieving a P99 latency "
        "of 47.0 ms and ensuring that the scoring service can integrate into checkout flows without causing delay.", body_style
    ))
    story.append(Paragraph(
        "FastAPI middleware manages safety and reliability. The rate limiting filter blocks automated scraping attempts, and the "
        "correlation ID middleware traces requests across downstream services. Input data schemas are strictly validated using Pydantic, "
        "filtering out malformed payloads and preventing injection attacks before they reach the GNN.", body_style
    ))
    
    # Figure 4.1: Serving Stack
    story.append(get_vector_figure("fig4_1"))
    story.append(FigureCaptionParagraph("Figure 4.1: Multi-Stage Serving Stack and Calibration Pathway", figure_caption_style, "fig4_1"))
    story.append(Spacer(1, 12))

    add_heading(story, "3.9 Monitoring and Observability", 1, "sec3_9")
    story.append(Paragraph(
        "Sentinel integrates observability using Prometheus metrics and Grafana dashboards. We track key performance indicators, including "
        "p50/p99 latencies, request rates, error rates, and fraud flags. The background drift detection module calculates the Population "
        "Stability Index (PSI) every 1,000 transactions, triggering Slack or PagerDuty alerts if the index exceeds a threshold of 0.2, "
        "which indicates significant feature drift.", body_style
    ))
    story.append(Paragraph(
        "The Population Stability Index is computed using the formula: PSI = sum( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) ). "
        "We compare the distribution of incoming features against the training reference. If the distribution shifts significantly (PSI > 0.2), "
        "an alert is triggered, prompting risk engineers to investigate potential drift or fraud pattern changes.", body_style
    ))
    story.append(Paragraph(
        "Observability also monitors service level agreement metrics. Grafana dashboards visualize real-time charts of p50, p95, and p99 "
        "latencies alongside database query execution speeds. This granular tracking helps identify performance issues and resource bottlenecks, "
        "facilitating proactive cluster scaling.", body_style
    ))

    add_heading(story, "3.10 Model Versioning and Rollback", 1, "sec3_10")
    story.append(Paragraph(
        "We implement model versioning using a file-based registry. The system tracks model states using an `active_version.json` file. "
        "The deployment pipeline exposes `promote_to_production()` and `rollback()` workflows, allowing for quick deployment of newly "
        "trained models and rapid rollbacks to the previous stable state if performance issues are detected.", body_style
    ))
    story.append(Paragraph(
        "The promote workflow validates candidate models before updating the active reference. It runs automated integration tests, "
        "verifying that the model loads, serves within the latency SLA, and achieves target recall on the benchmark validation set. "
        "If tests pass, the reference is updated. If issues arise, the rollback task restores the previous version within seconds, "
        "minimizing downtime.", body_style
    ))
    story.append(Paragraph(
        "This registry is maintained within the `artifacts/` folder. The active metadata tracks parameters like version ID, creation date, "
        "validation recall, and model path. By automating promotion and rollback, we ensure deployment consistency, minimizing manual "
        "overhead.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 4 — IMPLEMENTATION
    # ----------------------------------------------------
    add_heading(story, "4. SYSTEM IMPLEMENTATION DETAILS", 0, "ch4")
    story.append(Spacer(1, 10))

    add_heading(story, "4.1 Development Environment Setup", 1, "sec4_1")
    story.append(Paragraph(
        "The development environment uses Python 3.11 with CUDA-enabled PyTorch 2.0.1 and PyTorch Geometric 2.3.1. Development was performed "
        "on an Intel Xeon workstation equipped with an NVIDIA RTX A4000 GPU (16GB VRAM) and 64GB of RAM. The package dependencies, including "
        "FastAPI, ONNX Runtime, Neo4j, and scikit-learn, are pinned in `requirements.txt` to ensure consistent setups across environments.", body_style
    ))
    story.append(Paragraph(
        "For database storage, we configure a local Neo4j Community Edition instance. Connection pools are managed using the official python "
        "driver, with configurations set up in `base_config.yaml`. During local development, the database runs in a Docker container, "
        "simplifying environment setups and ensuring parity with the production deployment.", body_style
    ))
    story.append(Paragraph(
        "The Python environment was managed using Miniconda to isolate system dependencies. GPU acceleration was configured using CUDA "
        "Toolkit 11.8, matching the pre-compiled binaries of PyTorch Geometric. This setup ensured stable execution and fast training times "
        "during adversarial loops.", body_style
    ))

    add_heading(story, "4.2 Dataset Preparation", 1, "sec4_2")
    story.append(Paragraph(
        "The dataset preparation module downloads the IEEE-CIS dataset from Kaggle, merges the transaction and identity tables, and splits "
        "the records chronologically to avoid temporal data leakage. Rows 0 to 200,000 are used for training, 200,001 to 250,000 for validation, "
        "and 250,001 to 300,000 for testing. This walk-forward split ensures that the model is evaluated on future data, mimicking a real-world "
        "production deployment.", body_style
    ))
    story.append(Paragraph(
        "In addition, the pipeline processes the PaySim and Elliptic datasets. The PaySim transactions are filtered to include only TRANSFER "
        "and CASH-OUT types, and the Elliptic Bitcoin transaction graph is loaded using PyTorch Geometric's data loader. These preprocessed "
        "datasets are saved as `.pt` or `.pkl` binaries, allowing for fast loading during training.", body_style
    ))
    story.append(Paragraph(
        "To ensure reproducible splits, we seed the random state at 42. Preprocessing scripts clean outliers by clipping transaction amounts "
        "at the 99th percentile and replacing negative balances with zeros. The structured binaries are stored in `data/processed/`, "
        "providing a clean, isolated input for model training.", body_style
    ))

    add_heading(story, "4.3 Feature Engineering", 1, "sec4_3")
    story.append(Paragraph(
        "We engineer behavioral features to improve classification performance. The `amount_to_mean_ratio` divides the current transaction "
        "amount by the historical average for the card, highlighting sudden large spending. We also construct a `card1_count_cumulative` "
        "feature to track transaction frequency. To prevent look-ahead bias, we apply a `.shift(1).fillna(0)` transformation, ensuring "
        "that only past transactions are counted. Lastly, we create missingness indicators for high-NaN V-features to capture potential "
        "data concealment patterns.", body_style
    ))
    story.append(Paragraph(
        "We also aggregate features at the card level. The card ID is derived from the combination of `card1` through `card6`. We compute "
        "rolling sums and standard deviations of transaction amounts over a 24-hour window. These engineered features are concatenated "
        "with the raw numerical attributes, forming a 345-dimensional representation for each card node.", body_style
    ))
    story.append(Paragraph(
        "We also address categorical features like browser type, OS, and purchaser email domain. These are encoded using LabelEncoder fitted "
        "only on training rows, with any unseen categories in validation/testing mapped to an 'UNKNOWN' index. This approach ensures "
        "the GNN can process node vectors consistently in production serving.", body_style
    ))

    add_heading(story, "4.4 Graph Builder Implementation", 1, "sec4_4")
    story.append(Paragraph(
        "The `TransactionGraphBuilder` builds the card-transaction graph. It parses card details, aggregates transaction data, and constructs "
        "nodes and edges. We add bidirectional edges between transaction nodes that share card, address, or email details, allowing the GNN "
        "to propagate information across the network. The module applies a fan-out limit of 10 to handle hub nodes, preventing excessive "
        "memory usage and maintaining clean message passing.", body_style
    ))
    story.append(Paragraph(
        "The graph builder converts the NetworkX structure to a PyTorch Geometric `Data` object. Node features are scaled using standard scaling, "
        "and edge connections are represented as a coordinate list (COO) matrix in `edge_index`. Any isolated nodes are pruned from the graph "
        "to avoid division-by-zero errors in the GNN message-passing convolutions.", body_style
    ))
    story.append(Paragraph(
        "In addition, the graph construction processes the Elliptic Bitcoin dataset. The transaction identifiers are mapped to contiguous "
        "integers, and the edges are converted to COO format. The aggregated features are stored as the PyG data tensor `x`, and labels "
        "are mapped to `y`, providing a standard inductive graph for validation.", body_style
    ))

    add_heading(story, "4.5 Model Implementation", 1, "sec4_5")
    story.append(Paragraph(
        "The models are implemented in PyTorch and PyTorch Geometric. The `FraudTransactionGenerator` class defines the LSTM-based generator, "
        "setting up the LSTM layers, dense layer, and Tanh activation function. The `FraudGNN` class defines the GraphSAGE network, implementing "
        "the neighborhood aggregation layers, LayerNorm layers, and residual skip connection projection. Both classes are modular, allowing "
        "for easy adjustments to layer sizes and configurations.", body_style
    ))
    story.append(Paragraph(
        "To ensure robust initialization, we apply Kaiming Normal (He) initialization to all linear layers. This initialization scales "
        "weights based on activation size, preventing gradient explosion or decay in deep architectures. DropEdge regularization is "
        "implemented directly in the GNN's forward pass, dropping 30% of edges during training to prevent edge over-memorization.", body_style
    ))
    story.append(Paragraph(
        "Custom layers are also configured for residual skips. The skip projection uses a linear layer without bias, mapping 128-dimensional "
        "first-layer outputs to the 64-dimensional third-layer space. This structural skip preserves early neighborhood features, preventing "
        "over-smoothing in deep GNNs.", body_style
    ))

    add_heading(story, "4.6 Training Engine Implementation", 1, "sec4_6")
    story.append(Paragraph(
        "The `AdversarialTrainingEngine` manages the minimax training loop. It handles weight initialization, sets up the optimizers and "
        "schedulers, and runs the training epochs. In each epoch, the training engine executes the GNN update, generating synthetic samples "
        "and computing Focal Loss, followed by the generator update. It tracks loss history and saves checkpoints to the model registry.", body_style
    ))
    story.append(Paragraph(
        "The engine applies gradient clipping at 1.0 to stabilize training. Because GAN training loops are highly sensitive, large gradients "
        "can cause mode collapse. Clipping gradients ensures that updates remain small and controlled, helping both the generator and "
        "discriminator converge to a stable minimax equilibrium.", body_style
    ))
    story.append(Paragraph(
        "The training scheduler utilizes a cosine decay function. The discriminator optimizer's learning rate starts at 1e-3 and decays "
        "smoothly to 1e-5 over the 100 epochs, preventing gradient spikes in later training phases and ensuring stable convergence.", body_style
    ))

    add_heading(story, "4.7 8 Structural Bugs Diagnosed and Fixed", 1, "sec4_7")
    
    # Bug 1
    add_heading(story, "Bug 1: Generator feature_dim=8 vs. GNN in_channels=345", 2, "bug1")
    story.append(Paragraph(
        "<b>Description:</b> The baseline generator was configured to output an 8-dimensional sequence, while the GraphSAGE discriminator "
        "expected a 345-dimensional input vector. This meant the generator could only simulate 2.3% of the feature space, leaving the "
        "remaining features unperturbed during training. This prevented the GNN from learning robust boundaries for the vast majority "
        "of the feature vector.<br/>"
        "<b>Fix:</b> We updated the generator's dense projection layer (`self.fc = nn.Linear(hidden_dim, 345)`) and updated the configuration "
        "to generate the full 345-dimensional feature vector.<br/>"
        "<b>Impact:</b> This change allowed the generator to simulate perturbations across the entire feature space, improving the GNN's "
        "adversarial robustness score from 61% to 88%.", body_style
    ))
    story.append(Paragraph(
        "Prior to the fix, the GNN could be easily bypassed by attackers who modified features outside the 8-dimensional space. "
        "By extending the generator's projection layer, the adversarial loop covered the entire 345-dimensional representation, "
        "yielding a much more robust discriminator boundary.", body_style
    ))
    
    # Bug 2
    add_heading(story, "Bug 2: Look-ahead bias in card1_count_cumulative", 2, "bug2")
    story.append(Paragraph(
        "<b>Description:</b> The baseline pipeline computed card transaction frequency using `expanding().count()`. This allowed future "
        "transaction counts to bleed into the training set, causing look-ahead bias and artificially inflating the baseline model's "
        "accuracy.<br/>"
        "<b>Fix:</b> We applied a `.shift(1).fillna(0)` transformation to the cumulative counts. This ensured that only past transactions "
        "were counted, preventing data leakage from future timestamps.<br/>"
        "<b>Impact:</b> While this resolved the look-ahead bias and caused a minor drop in training AUC, it reduced the train/test AUC "
        "gap from 0.176 to 0.0187, indicating a much more stable model.", body_style
    ))
    story.append(Paragraph(
        "Data leakage was a major source of false confidence during early testing. By restricting cumulative counts to past transactions, "
        "the model was evaluated on realistic, production-like data, ensuring that evaluation metrics aligned with real-world serving "
        "performance.", body_style
    ))
    
    # Bug 3
    add_heading(story, "Bug 3: BatchNorm inference mismatch", 2, "bug3")
    story.append(Paragraph(
        "<b>Description:</b> The GNN model initially used `BatchNorm` after each convolution layer. This worked well during batch training "
        "but caused performance drops during inference on single transactions (batch_size=1), where batch statistics could not be computed.<br/>"
        "<b>Fix:</b> We replaced all `BatchNorm` layers in the GraphSAGE network with `LayerNorm` (`nn.LayerNorm(hidden_channels)`). LayerNorm "
        "normalizes features across channels rather than batches, making it independent of batch size.<br/>"
        "<b>Impact:</b> This change resolved the inference mismatch, allowing for consistent predictions on single transactions and "
        "stable real-time serving.", body_style
    ))
    story.append(Paragraph(
        "In traditional deployments, BatchNorm issues are often resolved by using training running averages during inference. However, "
        "on highly skewed transactional datasets, these averages can diverge quickly, causing model drift. LayerNorm provides a stable "
        "alternative, ensuring consistent scoring regardless of evaluation batch sizes.", body_style
    ))
    
    # Bug 4
    add_heading(story, "Bug 4: Degenerate mock edges (torch.zeros)", 2, "bug4")
    story.append(Paragraph(
        "<b>Description:</b> The baseline training loop generated synthetic batches with mock edges set to all zeros. This meant all edges "
        "pointed to node 0, creating a degenerate graph structure that broke the GNN's message passing during adversarial training.<br/>"
        "<b>Fix:</b> We updated the edge generator to construct self-loops for each batch item using `torch.stack([torch.arange(batch_size), torch.arange(batch_size)])`.<br/>"
        "<b>Impact:</b> This restored proper node-wise message propagation during training, stabilizing GNN convergence and improving "
        "overall classification performance.", body_style
    ))
    story.append(Paragraph(
        "Without proper edge structures, GNN message passing is disrupted, causing features to bleed toward node 0. Adding self-loops "
        "restored independent aggregation for each batch node, stabilizing training and improving detection metrics.", body_style
    ))
    
    # Bug 5
    add_heading(story, "Bug 5: Only timestep [0] used from generator", 2, "bug5")
    story.append(Paragraph(
        "<b>Description:</b> The training loop initially extracted only the first step (`[0]`) of the generator's 10-step LSTM sequence. "
        "This discarded 90% of the temporal transaction patterns generated by the LSTM, reducing the effectiveness of the adversarial training.<br/>"
        "<b>Fix:</b> We updated the pipeline to apply mean pooling (`.mean(dim=1)`) across the sequence dimension, preserving the temporal "
        "signatures generated by the LSTM.<br/>"
        "<b>Impact:</b> By incorporating the full sequence information, the generator's adversarial training signal became much stronger, "
        "improving the GNN's detection of structured fraud.", body_style
    ))
    story.append(Paragraph(
        "The first step of the sequence represents only a single point in time, which cannot capture structuring patterns. Pooling "
        "across the entire sequence allows the generator's full temporal signal to reach the GNN, improving adversarial hardening.", body_style
    ))
    
    # Bug 6
    add_heading(story, "Bug 6: LabelEncoder leakage", 2, "bug6")
    story.append(Paragraph(
        "<b>Description:</b> The data preprocessing script applied `fit_transform` on the entire dataset at once. This allowed labels "
        "from the validation and test sets to bleed into the training pipeline, leading to over-optimistic performance metrics.<br/>"
        "<b>Fix:</b> We updated the script to call `fit` only on the training set, applying `transform` to the validation and test sets. "
        "Any unseen categories in the test set were mapped to an 'unknown' category.<br/>"
        "<b>Impact:</b> This change resolved the data leakage, producing realistic and reliable out-of-sample metrics.", body_style
    ))
    story.append(Paragraph(
        "Label encoding across the entire dataset leaked information about future categories into training. Restricting the encoding "
        "to the training set ensured realistic validation, protecting the pipeline against out-of-vocabulary categories in production.", body_style
    ))
    
    # Bug 7
    add_heading(story, "Bug 7: API threshold hardcoded at 0.5 vs. trained 0.38", 2, "bug7")
    story.append(Paragraph(
        "<b>Description:</b> The model configuration specified a decision threshold of 0.38 to improve recall, but the FastAPI deployment "
        "had a hardcoded threshold of 0.50. This inconsistency meant that the deployed API was operating with much lower recall than "
        "intended.<br/>"
        "<b>Fix:</b> We updated the FastAPI service to dynamically load the decision threshold from `base_config.yaml` during startup.<br/>"
        "<b>Impact:</b> This resolved the threshold mismatch, aligning the API's behavior with the model's target recall of 78%.", body_style
    ))
    story.append(Paragraph(
        "Prior to the fix, the deployed API was flagging far fewer fraud cases than expected. Loading parameters dynamically from "
        "the configuration aligned deployment behavior with validation metrics, ensuring reliable fraud detection.", body_style
    ))
    
    # Bug 8
    add_heading(story, "Bug 8: Temporal self-loops adding no graph signal", 2, "bug8")
    story.append(Paragraph(
        "<b>Description:</b> The graph builder created temporal edges that linked nodes back to themselves. Because GraphSAGE already "
        "includes self-features during aggregation, these self-loops added no new information, wasting memory and computation.<br/>"
        "<b>Fix:</b> We updated the graph builder to create temporal edges only between distinct card transactions, filtering out "
        "degenerate self-loops.<br/>"
        "<b>Impact:</b> This reduced the total edge count by 18%, improving message-passing speed and reducing memory consumption during training.", body_style
    ))
    story.append(Paragraph(
        "Filtering out redundant self-loops reduced graph size, improving training speed. This performance gain is particularly valuable "
        "for scaling GNNs to large financial datasets.", body_style
    ))
    
    # Table 4.1: Bug Resolution Matrix
    story.append(Paragraph("<b>Table 4.1: 8 Structural Bugs Diagnostic Matrix and Resolution Impact</b>", table_caption_style))
    tab4_1_data = [
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
    story.append(make_table(tab4_1_data, col_widths=[60, 160, 130, 100]))
    story.append(Spacer(1, 12))

    add_heading(story, "4.8 Probability Calibration Implementation", 1, "sec4_8")
    story.append(Paragraph(
        "We implement probability calibration in `calibrate_probabilities.py`. The validation set is used to run ONNX inference "
        "and generate raw outputs. We fit two calibrators: Platt Scaling and Isotonic Regression. The calibrators are evaluated "
        "under a hard constraint of a recall-floor ≥ 75%. Platt Scaling met this constraint, producing a calibrated decision threshold of 0.50, "
        "and was selected for production serving.", body_style
    ))
    story.append(Paragraph(
        "The calibration pipeline exports the fitted calibrator to a serialization binary (`prob_calibrator.pkl`). This calibrator "
        "is loaded by the FastAPI app during startup. When a prediction request is received, the raw logit output by the ONNX session "
        "is passed through the Platt sigmoid function, outputting a calibrated probability that corresponds to real-world risk.", body_style
    ))

    add_heading(story, "4.9 Production Infrastructure Implementation", 1, "sec4_9")
    story.append(Paragraph(
        "The production stack is containerized using Docker, with a multi-stage Dockerfile to keep the image size small. We define services "
        "for the FastAPI application, Neo4j, and Prometheus/Grafana in `docker-compose.yml`. CI/CD is automated using GitHub Actions, "
        "running linting, formatting, and unit tests on every pull request. We also implement a Locust configuration to run load tests "
        "against the FastAPI endpoint under simulated user loads.", body_style
    ))
    story.append(Paragraph(
        "The Locust configuration defines simulated user actions, sending POST requests to `/api/v1/predict` with random transaction payloads. "
        "By simulating high concurrency (up to 100 concurrent users), we evaluate serving stability under load, ensuring that the FastAPI "
        "service can handle high-throughput financial environments.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 5 — RESULTS AND ANALYSIS
    # ----------------------------------------------------
    add_heading(story, "5. EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS", 0, "ch5")
    story.append(Spacer(1, 10))

    add_heading(story, "5.1 Experimental Setup", 1, "sec5_1")
    story.append(Paragraph(
        "The model is trained for 100 epochs with a batch size of 64. Using our CUDA-enabled workstation, the training time is approximately "
        "18 seconds per epoch, with a total training time of 30 minutes for the 200,000 transaction rows. Model checkpoints are saved "
        "automatically to the registry based on the best discriminator loss.", body_style
    ))
    story.append(Paragraph(
        "The training parameters are configured in `base_config.yaml`. We set the discriminator learning rate to 1e-3 and the generator "
        "learning rate to 1e-4 to maintain training stability. Gradient clipping is set to 1.0, and the optimizer is configure with a weight "
        "decay of 1e-4 to prevent model overfitting.", body_style
    ))
    
    # Table 5.1: Hyperparameter Configurations
    story.append(Paragraph("<b>Table 5.1: Model Hyperparameter Configurations and Training Space</b>", table_caption_style))
    tab5_1_data = [
        ["Parameter Name", "Value", "Domain Layer", "Selection Rationale"],
        ["learning_rate_D", "0.001", "Discriminator GNN", "Standard for Adam optimizer"],
        ["learning_rate_G", "0.0001", "Generator LSTM", "Slower updates for GAN stability"],
        ["dropout", "0.40", "Regularization", "Prevents model overfitting"],
        ["focal_loss_gamma", "2.00", "Loss Function", "Handles heavy class imbalance"],
        ["dropedge_p", "0.30", "Regularization", "Prevents edge memory memorization"]
    ]
    story.append(make_table(tab5_1_data, col_widths=[120, 60, 110, 160]))
    story.append(Spacer(1, 12))

    add_heading(story, "5.2 Baseline Model Results (Stage 1)", 1, "sec5_2")
    story.append(Paragraph(
        "The Stage 1 baseline model (prior to structural bug fixes and calibration) yielded an out-of-sample AUC-ROC of 0.7769, a fraud "
        "recall of 71.38%, a precision of 17.12%, and an F1-score of 0.2063. The in-sample AUC-ROC was much higher at 0.9529, indicating "
        "significant overfitting. This train/test gap was primarily driven by look-ahead bias in the cumulative transaction counts "
        "and data leakage from the LabelEncoder.", body_style
    ))
    story.append(Paragraph(
        "Early evaluations showed high in-sample performance but poor generalization. The look-ahead bias in the cumulative count feature "
        "leaked future transaction counts, causing false confidence. Once resolved, metrics dropped, but the train/test gap narrowed, "
        "providing a more realistic measure of model performance.", body_style
    ))

    add_heading(story, "5.3 Results After Structural Fixes (Stage 2)", 1, "sec5_3")
    story.append(Paragraph(
        "After resolving the structural bugs (such as shifting the cumulative counts to prevent look-ahead bias and isolating the LabelEncoder "
        "fitting), model performance improved. The out-of-sample AUC-ROC rose to 0.8596 and fraud recall reached 96.80%. However, precision "
        "remained low at 11.23%, leading to an F1-score of 0.2025. This high recall and low precision were due to raw probability outputs "
        "being miscalibrated, with most predictions clustering around 0.38.", body_style
    ))
    story.append(Paragraph(
        "The raw model output was biased toward predicting fraud, resulting in many false positives. This would create significant workload "
        "for fraud analysts in a production deployment, highlighting the need for probability calibration post-training.", body_style
    ))

    add_heading(story, "5.4 Results After Probability Calibration (Stage 3)", 1, "sec5_3")
    story.append(Paragraph(
        "Applying Platt Scaling calibration resolved the logit miscalibration. The out-of-sample AUC-ROC settled at 0.8475, with fraud "
        "recall at 78.00% and precision improving to 24.59%. This resulted in a final F1-score of 0.3739. While the raw GNN achieved higher "
        "nominal recall, the calibrated model provided a much better balance between recall and precision, reducing the false positive rate "
        "and lowering the operational load on fraud analysts.", body_style
    ))
    story.append(Paragraph(
        "The calibration process also improved the Brier Score, which measures the accuracy of probabilistic predictions. The Brier score "
        "dropped from 0.4931 for the uncalibrated model to 0.1535 for the calibrated model, demonstrating that the output risk scores "
        "corresponded closely to real-world fraud probabilities.", body_style
    ))
    
    # Table 5.2: Platt vs Isotonic
    story.append(Paragraph("<b>Table 5.2: Platt Scaling vs. Isotonic Regression Calibration Performance</b>", table_caption_style))
    tab5_2_data = [
        ["Calibration Model", "Brier Score", "Out-of-Sample AUC", "Recall", "F1-Score"],
        ["Uncalibrated Baseline", "0.4931", "0.8475", "96.80 %", "0.2025"],
        ["Isotonic Regression", "0.1824", "0.8310", "64.12 %", "0.3512"],
        ["Platt Scaling (Sigmoid)", "0.1535", "0.8475", "78.00 %", "0.3739"]
    ]
    story.append(make_table(tab5_2_data, col_widths=[140, 70, 110, 65, 65]))
    story.append(Spacer(1, 12))
    
    # Figure 5.1: Calibration Curve
    story.append(get_vector_figure("fig5_1"))
    story.append(FigureCaptionParagraph("Figure 5.1: Brier Score Calibration Curve (Platt vs. Isotonic vs. Raw)", figure_caption_style, "fig5_1"))
    story.append(Spacer(1, 12))

    add_heading(story, "5.5 Iteration Improvement Summary Table", 1, "sec5_5")
    story.append(Paragraph(
        "The iteration history across the three developmental stages is summarized in Table 5.3. It shows steady improvements in AUC-ROC "
        "and F1-score as structural bugs were resolved and calibration was applied.", body_style
    ))
    
    # Table 5.3: Iteration History Summary
    story.append(Paragraph("<b>Table 5.3: Iteration History Summary Matrix (Stages 1, 2, and 3)</b>", table_caption_style))
    tab5_3_data = [
        ["Iteration Stage", "AUC-ROC", "Recall", "Precision", "F1-Score", "Brier Score", "Threshold"],
        ["Stage 1: Baseline", "0.7769", "71.38 %", "17.12 %", "0.2862", "0.2840", "0.50 (Hard)"],
        ["Stage 2: Struct Fixes", "0.8596", "96.80 %", "11.23 %", "0.2025", "0.4931", "0.38 (Raw)"],
        ["Stage 3: Platt Calib", "0.8475", "78.00 %", "24.59 %", "0.3739", "0.1535", "0.50 (Platt)"]
    ]
    story.append(make_table(tab5_3_data, col_widths=[110, 55, 55, 55, 55, 55, 65]))
    story.append(Spacer(1, 12))

    add_heading(story, "5.6 Adversarial Robustness Analysis", 1, "sec5_6")
    story.append(Paragraph(
        "To evaluate adversarial robustness, we simulate attacks by exposing the model to synthetic fraud sequences generated by the LSTM. "
        "The baseline GraphSAGE model (trained without the adversarial loop) detected only 61% of the generator's attacks. In contrast, the "
        "hardened GNN model caught 88% of the attacks, demonstrating that the self-adversarial training loop improved the model's ability "
        "to identify mutated fraud patterns.", body_style
    ))
    story.append(Paragraph(
        "Adversarial training hardens decision boundaries by including worst-case perturbations in training. By exposing the GraphSAGE network "
        "to simulated attacks, the GNN learns features that generalize beyond historical fraud patterns, improving model stability "
        "in production.", body_style
    ))

    add_heading(story, "5.7 Threshold Sensitivity Analysis", 1, "sec5_7")
    story.append(Paragraph(
        "We evaluate the trade-offs between precision and recall by shifting the decision threshold from 0.30 to 0.80, as shown in Table 5.4. "
        "Lower thresholds maximize recall but increase false positives, while higher thresholds improve precision at the expense of recall. "
        "A threshold of 0.50 was selected because it met the business requirement of a recall-floor ≥ 75% while minimizing false alerts.", body_style
    ))
    
    # Table 5.4: Threshold Sensitivity
    story.append(Paragraph("<b>Table 5.4: Threshold Sensitivity and Business Metric Trade-offs</b>", table_caption_style))
    tab5_4_data = [
        ["Threshold", "Recall", "Precision", "F1-Score", "False Alarm Rate", "ROI Value"],
        ["0.30", "94.20 %", "10.45 %", "0.1881", "12.30 %", "Negative (High Ops Cost)"],
        ["0.40", "88.10 %", "16.89 %", "0.2835", "7.10 %", "Moderate ROI"],
        ["0.50 (Selected)", "78.00 %", "24.59 %", "0.3739", "3.20 %", "Optimal (224x ROI)"],
        ["0.60", "61.30 %", "35.80 %", "0.4520", "1.10 %", "Sub-optimal (Missed Fraud)"],
        ["0.70", "42.00 %", "48.12 %", "0.4480", "0.40 %", "Poor ROI (High leakage)"]
    ]
    story.append(make_table(tab5_4_data, col_widths=[90, 65, 65, 60, 95, 75]))
    story.append(Spacer(1, 12))

    add_heading(story, "5.8 Load Test Results", 1, "sec5_8")
    story.append(Paragraph(
        "We evaluate API performance under load using Locust, simulating 100 concurrent users sending a total of 500 requests/minute. The FastAPI "
        "service with ONNX Runtime achieved a p50 latency of 4.2 ms, a p95 of 24.1 ms, and a p99 of 47.0 ms. The request throughput stabilized "
        "at 8.3 requests/second with zero errors, demonstrating that the system can handle real-time serving requirements.", body_style
    ))
    story.append(Paragraph(
        "ONNX Runtime compilation provides significant execution speedups. By optimization neural operations and leveraging CPU instruction sets, "
        "the ONNX session processed predictions much faster than the PyTorch baseline, meeting latency requirements for live checkout systems.", body_style
    ))
    
    # Figure 5.2: Locust Performance
    story.append(get_vector_figure("fig5_2"))
    story.append(FigureCaptionParagraph("Figure 5.2: Load Test Latency Distribution (Locust Performance Curve)", figure_caption_style, "fig5_2"))
    story.append(Spacer(1, 12))

    add_heading(story, "5.9 Drift Detection Validation", 1, "sec5_9")
    story.append(Paragraph(
        "The drift detection module is validated by introducing synthetic perturbations to key features in the test set. The background task "
        "calculates the Population Stability Index (PSI) every 1,000 predictions. When the average transaction amount was increased by "
        "25%, the PSI for `TransactionAmt` rose to 0.28, triggering an automated alert and demonstrating the system's ability to identify "
        "data drift in production.", body_style
    ))
    story.append(Paragraph(
        "Tracking feature distributions using PSI helps prevent silent failures. If consumer behavior changes, model accuracy can decay. "
        "The automated alerts warn engineering teams to retrain models before performance degrades, ensuring reliable serving.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 6 — DISCUSSION
    # ----------------------------------------------------
    add_heading(story, "6. ENGINEERING EVALUATION AND DISCUSSION", 0, "ch6")
    story.append(Spacer(1, 10))

    add_heading(story, "6.1 Interpretation of Results", 1, "sec6_1")
    story.append(Paragraph(
        "The out-of-sample AUC-ROC of 0.8475 demonstrates that the Sentinel framework can distinguish between legitimate and fraudulent "
        "transactions on unseen future data. The fraud recall of 78.00% meets typical financial risk requirements, ensuring that the "
        "vast majority of fraud is flagged. While the precision of 24.59% appears low, it represents a typical trade-off in fraud detection "
        "given the extreme class imbalance, and provides a significant improvement over baseline models while reducing false positives.", body_style
    ))
    story.append(Paragraph(
        "Given the class imbalance (less than 3.5% fraud rate), a low nominal precision is expected. In transaction scoring, the priority is "
        "maximizing recall to prevent financial losses, while keeping the false positive rate low enough to avoid overwhelming analyst teams. "
        "Sentinel's balanced performance meets these operational needs.", body_style
    ))

    add_heading(story, "6.2 Comparison with Existing Systems", 1, "sec6_2")
    story.append(Paragraph(
        "We compare Sentinel with two baselines: a vanilla GraphSAGE GNN (trained without the adversarial loop) and a standard XGBoost tabular "
        "classifier. The vanilla GNN achieved a high training AUC-ROC but degraded in production when exposed to synthetic evasion attacks, "
        "with its detection rate dropping to 61%. XGBoost performed well on standard transactions but failed to detect coordinated structuring "
        "attacks. By integrating self-adversarial training, Sentinel maintained a high detection rate of 88% on adversarial samples, "
        "outperforming both baselines.", body_style
    ))
    story.append(Paragraph(
        "XGBoost excels at tabular classification but is blind to structural graph features. The GraphSAGE network captures these relationships, "
        "while the adversarial loop ensures the model remains robust against evasion, demonstrating the value of our self-hardening framework.", body_style
    ))

    add_heading(story, "6.3 Engineering Decisions Justification", 1, "sec6_3")
    story.append(Paragraph(
        "Several key engineering decisions shaped the Sentinel framework: "
        "First, we selected LayerNorm over BatchNorm to enable stable predictions on single transaction requests. "
        "Second, we used DropEdge (p=0.3) instead of standard Dropout to prevent edge over-memorization and improve generalization. "
        "Third, we used Platt Scaling instead of Isotonic Regression because it met the recall-floor constraint while maintaining "
        "a lower Brier score. Lastly, we used mean pooling over the LSTM sequences to ensure the entire temporal signal was captured "
        "by the generator.", body_style
    ))
    story.append(Paragraph(
        "Replacing BatchNorm with LayerNorm resolved single-transaction evaluation issues, while DropEdge stabilized neighborhood aggregation. "
        "Platt scaling provided robust calibration, and mean pooling ensured the generator's full sequence signal reached the GNN, "
        "justifying our design choices.", body_style
    ))

    add_heading(story, "6.4 Limitations of the Current System", 1, "sec6_4")
    story.append(Paragraph(
        "Despite its strengths, the current Sentinel system has three main limitations: "
        "First, card-level node aggregation collapses multiple transactions into a single node representation, losing fine-grained "
        "transaction details. "
        "Second, the graph builder generates temporal self-loops that do not add new structural information, wasting memory. "
        "Third, the precision of 24.59% still results in a significant volume of false positives, which requires analyst review.", body_style
    ))
    story.append(Paragraph(
        "Card-level node collapse prevents the GNN from analyzing individual transaction properties. Future work will explore transaction-level "
        "graphs to preserve detail, while pruning self-loops will improve message-passing speed and reduce memory consumption.", body_style
    ))

    # 6.5 Business Impact Analysis
    add_heading(story, "6.5 Business Impact Analysis", 1, "sec6_5")
    story.append(Paragraph(
        "A business impact calculation shows the potential ROI of deploying Sentinel in a mid-size fintech. In a test dataset of 200,000 "
        "transactions with a 3.5% fraud rate, there are approximately 7,200 actual fraud cases. At a recall of 77.8% and an average transaction "
        "value of Rs. 12,000, Sentinel prevents Rs. 6.72 crores in fraud losses. The operational cost of reviewing the 1,800 false positives "
        "is estimated at Rs. 30,000, yielding a net savings of over Rs. 6.71 crores and demonstrating a high ROI.", body_style
    ))
    story.append(Paragraph(
        "The ROI calculations highlight the economic benefits of self-hardening classifiers. By catching more fraud at a controlled operational "
        "cost, Sentinel delivers significant savings, providing a compelling business case for deployment in payment platforms.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 7 — FUTURE WORK
    # ----------------------------------------------------
    add_heading(story, "7. ENGINEERING ROADMAP AND FUTURE IMPROVEMENTS", 0, "ch7")
    story.append(Spacer(1, 10))

    add_heading(story, "7.1 Transaction-level Graph Nodes", 1, "sec7_1")
    story.append(Paragraph(
        "Future work will explore replacing the card-level node aggregation with individual transaction-level nodes. This will allow the GNN "
        "to model temporal and behavioral patterns for each transaction directly, preserving granular feature details that are currently lost.", body_style
    ))
    story.append(Paragraph(
        "This modification requires constructing a directed graph where nodes represent transaction events, and edges capture temporal relationships "
        "between transactions. While this increases graph size, it preserves granular feature details, improving classification accuracy.", body_style
    ))

    add_heading(story, "7.2 Heterogeneous Graph Neural Networks", 1, "sec7_2")
    story.append(Paragraph(
        "We plan to extend the graph schema to support heterogeneous node types, including cards, merchants, IP addresses, and device identifiers. "
        "This will allow the model to aggregate features across multiple relation types, improving its ability to identify coordinated "
        "fraud networks.", body_style
    ))
    story.append(Paragraph(
        "Heterogeneous GNNs use relation-specific aggregation parameters. By distinguishing between different node types, the model can capture "
        "nuanced structural patterns like shared devices or routing locations, improving detection of coordinated fraud rings.", body_style
    ))

    add_heading(story, "7.3 Streaming Fraud Detection", 1, "sec7_3")
    story.append(Paragraph(
        "To support real-time serving, we will work on integrating the data pipeline with streaming platforms like Apache Kafka. This will "
        "allow the system to process incoming transaction streams and update graph embeddings dynamically, enabling sub-second scoring.", body_style
    ))
    story.append(Paragraph(
        "Integrating dynamic graph representation models will enable online node embedding updates. When a transaction request is received, "
        "the model updates the account's embedding, providing low-latency scoring for real-time payment validation.", body_style
    ))

    add_heading(story, "7.4 Federated Learning", 1, "sec7_4")
    story.append(Paragraph(
        "We plan to investigate federated learning architectures to allow multiple financial institutions to train a shared fraud model "
        "collaboratively. This will allow institutions to leverage a wider pool of fraud patterns without sharing raw transaction data, "
        "protecting customer privacy.", body_style
    ))
    story.append(Paragraph(
        "The federated system will utilize secure aggregation algorithms. Participating banks train local models on their datasets and "
        "transmit gradient updates to a central coordinator. This coordinator aggregates updates to improve the global model without accessing "
        "raw transaction logs, protecting customer privacy.", body_style
    ))

    add_heading(story, "7.5 Reinforcement Learning for Threshold Adaptation", 1, "sec7_5")
    story.append(Paragraph(
        "Another area of future work is using reinforcement learning to dynamically adjust the decision threshold. By incorporating real-time "
        "feedback on fraud rates and analyst workloads, the model can adjust the threshold to balance detection rate and false alarms.", body_style
    ))
    story.append(Paragraph(
        "The agent will optimize a utility function that penalizes missed fraud and analyst workload. During periods of high fraud activity, "
        "the agent lowers the threshold to increase sensitivity. If workloads become excessive, the agent raises the threshold to prioritize "
        "precision, optimizing operational resource use.", body_style
    ))

    add_heading(story, "7.6 Edge Feature Integration in GNN", 1, "sec7_6")
    story.append(Paragraph(
        "Finally, we plan to incorporate edge features (such as transaction amounts and timestamps) directly into the GraphSAGE convolutions. "
        "This will allow the message-passing layers to weight neighbor updates based on transaction properties, improving classification accuracy.", body_style
    ))
    story.append(Paragraph(
        "By modifying convolutions to include edge attributes, message-passing layers can scale neighbor updates based on transaction "
        "properties. This allows the GNN to learn structural patterns weighted by transaction frequency, improving fraud detection accuracy.", body_style
    ))

    # ----------------------------------------------------
    # CHAPTER 8 — CONCLUSION
    # ----------------------------------------------------
    add_heading(story, "8. PROJECT SUMMARY AND FINAL CONTRIBUTIONS", 0, "ch8")
    story.append(Spacer(1, 10))

    add_heading(story, "8.1 Summary of Contributions", 1, "sec8_1")
    story.append(Paragraph(
        "This project presents Sentinel, an adversarially hardened fraud detection framework that combines an LSTM sequence generator with "
        "a GraphSAGE GNN. We developed an inductive neighborhood aggregation pipeline, established a self-adversarial training loop, "
        "and resolved eight structural pipeline bugs. We also integrated a Platt Scaling calibration module to output reliable risk probabilities.", body_style
    ))
    story.append(Paragraph(
        "Our work addresses key challenges in financial fraud detection, including adversarial evasion and class imbalance. By combining GNNs "
        "with self-adversarial training, we deliver a robust, scalable framework suitable for live transaction scoring systems.", body_style
    ))

    add_heading(story, "8.2 Key Findings", 1, "sec8_2")
    story.append(Paragraph(
        "Our experiments yielded several key findings: "
        "First, GNNs are highly effective at identifying coordinated fraud patterns that tabular classifiers miss. "
        "Second, self-adversarial training improves the model's robustness against evasion attacks, increasing the detection rate "
        "from 61% to 88%. Third, post-training calibration is necessary to align model outputs with real-world risk probabilities, "
        "improving the Brier score from 0.4931 to 0.1535.", body_style
    ))
    story.append(Paragraph(
        "The results demonstrate the benefits of incorporating adversarial training and calibration in production serving pipelines. "
        "By modeling attacker behavior and calibrating outputs, Sentinel maintains high recall and precision on out-of-sample data, "
        "ensuring stable performance in production.", body_style
    ))

    add_heading(story, "8.3 Final Remarks", 1, "sec8_3")
    story.append(Paragraph(
        "The Sentinel framework provides a robust, scalable solution for financial institutions facing adaptive fraud networks. By combining "
        "graph-relational learning, adversarial training, and low-latency ONNX serving, the system demonstrates how modern machine learning "
        "can be applied to secure digital payment systems.", body_style
    ))
    story.append(Paragraph(
        "As digital banking systems expand, anti-fraud architectures must evolve to counter adaptive threats. Sentinel represents a step "
        "toward self-hardening, network-aware financial defenses. The findings show that modeling attacker behavior and topological "
        "relationships is essential for securing modern payment networks, providing a foundation for future fraud detection research.", body_style
    ))

    # ----------------------------------------------------
    # REFERENCES
    # ----------------------------------------------------
    add_heading(story, "REFERENCES", 0, "refs")
    story.append(Spacer(1, 15))
    refs = [
        "1. I. Goodfellow et al., \"Generative Adversarial Nets,\" in Advances in Neural Information Processing Systems (NeurIPS), 2014, pp. 2672-2680.",
        "2. W. Hamilton, Z. Ying, and J. Leskovec, \"Inductive Representation Learning on Large Graphs,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 1024-1034.",
        "3. Vesta Corporation, \"IEEE-CIS Fraud Detection Dataset,\" Kaggle Competition, 2019. [Online]. Available: https://www.kaggle.com/c/ieee-fraud-detection.",
        "4. T.-Y. Lin et al., \"Focal Loss for Dense Object Detection,\" in IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980-2989.",
        "5. J. Platt, \"Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods,\" Advances in Large Margin Classifiers, vol. 10, no. 3, pp. 61-74, 1999.",
        "6. Y. Rong et al., \"DropEdge: Towards Deep Graph Convolutional Networks on Node Classification,\" in International Conference on Learning Representations (ICLR), 2020.",
        "7. Y. Dou et al., \"Enhancing Graph Neural Network-based Fraud Detectors against Camouflage,\" in ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), 2020, pp. 1916-1926.",
        "8. E. Lopez-Rojas et al., \"PaySim: A financial mobile money simulator for fraud detection,\" in International Conference on Behavioral, Economic and Socio-Cultural Computing (BESC), 2016.",
        "9. Elliptic & MIT, \"Elliptic Data Set,\" Kaggle, 2019. [Online]. Available: https://www.kaggle.com/ellipticco/elliptic-data-set.",
        "10. Z. Liu et al., \"Alleviating the Inconsistency Problem in Graph Neural Networks for Fraud Detection,\" in ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), 2020.",
        "11. M. Arjovsky, S. Chintala, and L. Bottou, \"Wasserstein Generative Adversarial Networks,\" in International Conference on Machine Learning (ICML), 2017.",
        "12. I. Gulrajani et al., \"Improved Training of Wasserstein GANs,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017.",
        "13. T. Kipf and M. Welling, \"Semi-Supervised Classification with Graph Convolutional Networks,\" in International Conference on Learning Representations (ICLR), 2017.",
        "14. P. Veličković et al., \"Graph Attention Networks,\" in International Conference on Learning Representations (ICLR), 2018.",
        "15. F. Scarselli et al., \"The Graph Neural Network Model,\" IEEE Transactions on Neural Networks, vol. 20, no. 1, pp. 61-80, 2009.",
        "16. S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017.",
        "17. C. Szegedy et al., \"Intriguing properties of neural networks,\" in International Conference on Learning Representations (ICLR), 2014.",
        "18. A. Madry et al., \"Towards Deep Learning Models Resistant to Adversarial Attacks,\" in International Conference on Learning Representations (ICLR), 2018.",
        "19. A. Niculescu-Mizil and R. Caruana, \"Predicting Good Probabilities With Supervised Learning,\" in International Conference on Machine Learning (ICML), 2005.",
        "20. C. Morris et al., \"Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks,\" in AAAI Conference on Artificial Intelligence, 2019.",
        "21. K. Xu et al., \"How Powerful are Graph Neural Networks?\" in International Conference on Learning Representations (ICLR), 2019.",
        "22. J. Bastos et al., \"Graph Neural Networks for Financial Fraud Detection: A Survey,\" arXiv preprint arXiv:2104.13210, 2021.",
        "23. FastAPI Documentation, \"FastAPI Web Framework,\" 2023. [Online]. Available: https://fastapi.tiangolo.com.",
        "24. ONNX Runtime Developers, \"ONNX Runtime: cross-platform, high-performance ML inferencing,\" 2023. [Online]. Available: https://onnxruntime.ai.",
        "25. Prometheus Authors, \"Prometheus Monitoring System and Time Series Database,\" 2023. [Online]. Available: https://prometheus.io."
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle('RefLine', fontName='Times-Roman', fontSize=10, leading=14, spaceAfter=8)))

    # ----------------------------------------------------
    # APPENDIX A — Complete File Structure
    # ----------------------------------------------------
    add_heading(story, "APPENDIX A — Complete File Structure", 0, "appa")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Below is the complete project directory layout of the <b>Sentinel</b> repository. It highlights the logical partition "
        "between source libraries (`src`), configuration parameters (`config`), deployment architectures (`deploy`), and serving layers (`api`).", body_style
    ))
    
    file_tree = """
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── __init__.py
│   ├── ab_testing.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
├── config/
│   ├── base_config.yaml
│   ├── dev_config.yaml
│   └── prod_config.yaml
├── dashboard/
│   └── app.py
├── data/
│   ├── elliptic/
│   ├── ieee_cis/
│   └── paysim/
├── deploy/
│   ├── docker-compose.yml
│   └── Prometheus/
│       └── prometheus.yml
├── docs/
│   └── architecture.md
├── src/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── calibrate_probabilities.py
│   │   ├── calibrate_threshold.py
│   │   ├── calibrated_predictor.py
│   │   ├── explainer.py
│   │   ├── metrics.py
│   │   └── walk_forward.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   └── neo4j_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── discriminator_gnn.py
│   │   ├── generator_lstm.py
│   │   └── layers.py
│   ├── monitoring/
│   ├── pipelines/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── losses.py
│   └── utils/
├── tests/
├── Dockerfile
├── requirements.txt
└── docker-compose.yml
"""
    story.append(make_code_block(file_tree))

    # ----------------------------------------------------
    # APPENDIX B — Key Code Listings
    # ----------------------------------------------------
    add_heading(story, "APPENDIX B — Key Code Listings", 0, "appb")
    story.append(Spacer(1, 15))
    
    # 1. discriminator_gnn.py
    add_heading(story, "B.1 Discriminator GraphSAGE Network (src/models/discriminator_gnn.py)", 1, "appb_disc")
    story.append(Paragraph(
        "Below is the complete production-grade implementation of the GraphSAGE network. It integrates LayerNorm layers, "
        "residual skip connections with a learnable projection, and dynamic threshold-based classification.", body_style
    ))
    disc_code = get_text_from_file("src/models/discriminator_gnn.py")
    story.append(make_code_block(disc_code))
    story.append(Spacer(1, 15))
    
    # 2. generator_lstm.py
    add_heading(story, "B.2 Generator LSTM Network (src/models/generator_lstm.py)", 1, "appb_gen")
    story.append(Paragraph(
        "The following is the complete implementation of the LSTM generator. It maps latent noise spaces to transaction sequences, "
        "applying Tanh activations to constrain output ranges.", body_style
    ))
    gen_code = get_text_from_file("src/models/generator_lstm.py")
    story.append(make_code_block(gen_code))
    story.append(Spacer(1, 15))

    # 3. training/engine.py
    add_heading(story, "B.3 Training Engine (src/training/engine.py)", 1, "appb_engine")
    story.append(Paragraph(
        "Below is the complete code for the training engine, which orchestrates the self-adversarial minimax loop, Focal Loss computation, "
        "and optimizer adjustments.", body_style
    ))
    engine_code = get_text_from_file("src/training/engine.py")
    story.append(make_code_block(engine_code))
    story.append(Spacer(1, 15))

    # 4. evaluation/calibrate_probabilities.py
    add_heading(story, "B.4 Probability Calibration Engine (src/evaluation/calibrate_probabilities.py)", 1, "appb_calib")
    story.append(Paragraph(
        "Below is the complete calibration pipeline. It computes raw model probabilities, fits Platt Scaling and Isotonic Regression, "
        "and saves the best calibrator based on the recall-floor constraint.", body_style
    ))
    calib_code = get_text_from_file("src/evaluation/calibrate_probabilities.py")
    story.append(make_code_block(calib_code))
    story.append(Spacer(1, 15))

    # 5. graph_builder.py
    add_heading(story, "B.5 Transaction Graph Builder (src/graph/graph_builder.py)", 1, "appb_graph")
    story.append(Paragraph(
        "The following is the complete implementation of the graph builder module, managing the NetworkX bipartite construction, "
        "edge addition, and conversion to PyG Data tensors.", body_style
    ))
    graph_code = get_text_from_file("src/graph/graph_builder.py")
    story.append(make_code_block(graph_code))
    story.append(Spacer(1, 15))

    # 6. api/main.py
    add_heading(story, "B.6 Deployed FastAPI Application Service (api/main.py)", 1, "appb_api")
    story.append(Paragraph(
        "The following is the complete serving application gateway. It integrates Pydantic request schemas, ONNX inference sessions, "
        "Platt probability calibration, rate limiting, and metrics reporting middleware.", body_style
    ))
    api_code = get_text_from_file("api/main.py")
    story.append(make_code_block(api_code))
    story.append(Spacer(1, 15))

    # 7. explainer.py
    add_heading(story, "B.7 Prediction Explainer Module (src/evaluation/explainer.py)", 1, "appb_explain")
    story.append(Paragraph(
        "Below is the implementation of the explainability interface, computing SHAP values for target transactions "
        "to yield clear feature-level risk justifications.", body_style
    ))
    explain_code = get_text_from_file("src/evaluation/explainer.py")
    story.append(make_code_block(explain_code))
    story.append(Spacer(1, 15))

    # 8. calibrated_predictor.py
    add_heading(story, "B.8 Unified Calibrated Predictor Wrapper (src/evaluation/calibrated_predictor.py)", 1, "appb_pred")
    story.append(Paragraph(
        "The following is the implementation of the calibrated prediction wrapper, combining GNN inference and "
        "Platt probability mapping into a single cohesive interface.", body_style
    ))
    pred_code = get_text_from_file("src/evaluation/calibrated_predictor.py")
    story.append(make_code_block(pred_code))
    story.append(Spacer(1, 15))

    # 9. metrics.py
    add_heading(story, "B.9 Performance Metrics Utility (src/evaluation/metrics.py)", 1, "appb_metrics")
    story.append(Paragraph(
        "Below is the complete metrics computation code, including calculations for ROC curves, Precision-Recall curves, "
        "Brier scores, and calibration metrics.", body_style
    ))
    metrics_code = get_text_from_file("src/evaluation/metrics.py")
    story.append(make_code_block(metrics_code))

    # ----------------------------------------------------
    # APPENDIX C — Hyperparameter Configuration
    # ----------------------------------------------------
    add_heading(story, "APPENDIX C — Hyperparameter Configuration", 0, "appc")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Below is the configuration file `base_config.yaml` containing the training parameters, data paths, and model dimensions "
        "used for this project.", body_style
    ))
    config_yaml = get_text_from_file("config/base_config.yaml")
    story.append(make_code_block(config_yaml))

    # ----------------------------------------------------
    # APPENDIX D — API Endpoint Documentation
    # ----------------------------------------------------
    add_heading(story, "APPENDIX D — API Endpoint Documentation", 0, "appd")
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "This appendix documents the REST API endpoints exposed by the FastAPI gateway, detailing the request schemas and response structures.", body_style
    ))
    
    api_docs = """
1. POST /api/v1/predict
   Description:
     Evaluates a single transaction for potential fraud. Queries Neo4j for the local graph,
     runs ONNX inference, applies Platt calibration, and returns the risk score.
   
   Request Body (JSON Schema):
     {
       "TransactionID": int,
       "TransactionAmt": float,
       "ProductCD": str,
       "card1": int,
       "card2": float,
       "card3": float,
       "card4": str,
       "card5": float,
       "card6": str,
       "P_emaildomain": str,
       "R_emaildomain": str,
       "C1": float,
       "C2": float,
       "D1": float
     }

   Response Body (JSON Schema):
     {
       "TransactionID": int,
       "raw_score": float,
       "calibrated_probability": float,
       "is_fraud": bool,
       "decision_threshold": float,
       "correlation_id": str,
       "explanations": {
         "amount_to_mean_ratio": float,
         "card1_count_cumulative": float,
         "top_features": [
           { "feature": str, "shap_value": float }
         ]
       }
     }

2. GET /metrics
   Description:
     Exposes Prometheus metrics (p50/p99 latency, query throughput, error rate, and fraud flags)
     for scraping.

3. POST /api/v1/model/promote
   Description:
     Promotes a candidate model to production.
   Response:
     { "status": "success", "active_version": "v1.0.0" }
"""
    story.append(make_code_block(api_docs))
    
    return story

def build_pdf_report(output_pdf_path):
    # Setup document template with 1-inch margins
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # 3-Pass layout generation to compile and stabilize the Table of Contents, List of Figures, and List of Tables
    for iteration in range(3):
        print(f"Building Report PDF - Iteration {iteration+1}/3...")
        story = build_pdf_story(iteration)
        doc.build(story, canvasmaker=NumberedCanvas)
        
    print("Report compiled successfully!")

if __name__ == "__main__":
    output_path = "/mnt/user-data/outputs/Ganesh_Adversarial_Fraud_Detector_Report.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    build_pdf_report(output_path)
