"""
Exports reports/technical_review_report.md into a fully formatted Microsoft Word (.docx) document:
reports/technical_review_report.docx
Includes:
- Academic formatting (margins, headings, fonts, table styling)
- Beautifully formatted mathematical equation callout boxes with clean Unicode symbols
- Formatted tables with headers and borders
- High-resolution embedded figures with captions from results/report_assets/
"""

import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPORT_MD = "reports/technical_review_report.md"
REPORT_DOCX = "reports/technical_review_report.docx"
ASSET_DIR = "results/report_assets"


def set_cell_background(cell, fill_hex):
    """Sets the background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def clean_latex_to_unicode(text):
    """Converts LaTeX math syntax into clean readable Unicode mathematical notation."""
    if not text:
        return ""
    t = text.strip()
    
    # Cases environment
    if r'\begin{cases}' in t:
        m = re.search(r'\\begin\{cases\}(.*?)\\end\{cases\}', t, re.DOTALL)
        if m:
            cases_content = m.group(1).strip()
            branches = [b.strip() for b in cases_content.split(r'\\')]
            prefix = t[:t.find(r'\begin{cases}')].strip()
            
            cleaned_branches = []
            for idx, b in enumerate(branches):
                parts = [p.strip() for p in b.split('&')]
                clean_parts = [clean_latex_to_unicode(p) for p in parts]
                line_str = "    ".join(clean_parts)
                delim = "⎧ " if idx == 0 else ("⎩ " if idx == len(branches)-1 else "⎨ ")
                cleaned_branches.append(f"  {delim}{line_str}")
                
            prefix_clean = clean_latex_to_unicode(prefix)
            if prefix_clean:
                return f"{prefix_clean}\n" + "\n".join(cleaned_branches)
            return "\n".join(cleaned_branches)

    # Common LaTeX replacements
    replacements = [
        (r'\\mathcal\{D\}', '𝒟'),
        (r'\\mathcal\{X\}', '𝒳'),
        (r'\\mathcal\{U\}', '𝒰'),
        (r'\\mathcal\{L\}', 'ℒ'),
        (r'\\mathbb\{R\}\^d', 'ℝᵈ'),
        (r'\\mathbb\{R\}', 'ℝ'),
        (r'\\mathbb\{I\}', '𝕀'),
        (r'\\mathbf\{x\}_i', '𝐱ᵢ'),
        (r'\\mathbf\{x\}_j', '𝐱ⱼ'),
        (r'\\mathbf\{x\}_k', '𝐱ₖ'),
        (r'\\mathbf\{x\}_{zi}', '𝐱_zi'),
        (r'\\mathbf\{x\}_{new}', '𝐱_new'),
        (r'\\mathbf\{x\}', '𝐱'),
        (r'\\mathbf\{X\}', '𝐗'),
        (r'\\mathbf\{LS\}', '𝐋𝐒'),
        (r'\\hat\{y\}_i', 'ŷᵢ'),
        (r'\\hat\{y\}', 'ŷ'),
        (r'y_i', 'yᵢ'),
        (r'x_i', 'xᵢ'),
        (r'p_i', 'pᵢ'),
        (r'_\{i=1\}\^\{N\}', '  (for i = 1, ..., N)'),
        (r'\\Delta I_G', 'ΔI_G'),
        (r'\\Delta', 'Δ'),
        (r'\\sum', 'Σ'),
        (r'\\ge', '≥'),
        (r'\\le', '≤'),
        (r'\\in', '∈'),
        (r'\\forall', '∀'),
        (r'\\setminus', '∖'),
        (r'\\rightarrow', '→'),
        (r'\\times', '×'),
        (r'\\tau', 'τ'),
        (r'\\beta', 'β'),
        (r'\\sim', '~'),
        (r'\\mid', '|'),
        (r'\\quad', ' '),
        (r'\\text\{([^}]*)\}', r'\1'),
        (r'\\\{', '{'),
        (r'\\\}', '}'),
        (r'\\%', '%'),
    ]
    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t)
        
    return t


def add_formatted_text(paragraph, text, bold=False, italic=False, color=None, font_size=11):
    """Adds styled runs to paragraph, parsing bold **text**, code `code`, and inline math $math$."""
    # Protect escaped dollar signs (currency like \$500, \$10, \$2,230)
    text = text.replace(r'\$', '___DOLLARSIGN___')

    # Split by bold tags
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            sub_text = part[2:-2]
            # Recursively format within bold or add bold runs
            sub_parts = re.split(r'(`.*?`|\$.*?\$)', sub_text)
            for sp in sub_parts:
                if sp.startswith('`') and sp.endswith('`'):
                    run = paragraph.add_run(sp[1:-1].replace('___DOLLARSIGN___', '$'))
                    run.font.name = 'Consolas'
                    run.font.size = Pt(font_size - 1)
                    run.bold = True
                elif sp.startswith('$') and sp.endswith('$') and len(sp) > 2:
                    math_clean = clean_latex_to_unicode(sp[1:-1]).replace('___DOLLARSIGN___', '$')
                    m_run = paragraph.add_run(math_clean)
                    m_run.italic = True
                    m_run.bold = True
                    m_run.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
                else:
                    run = paragraph.add_run(sp.replace('___DOLLARSIGN___', '$'))
                    run.bold = True
                    if color:
                        run.font.color.rgb = color
                    if font_size:
                        run.font.size = Pt(font_size)
        else:
            # Check for inline code `code`
            sub_parts = re.split(r'(`.*?`)', part)
            for sp in sub_parts:
                if sp.startswith('`') and sp.endswith('`'):
                    run = paragraph.add_run(sp[1:-1].replace('___DOLLARSIGN___', '$'))
                    run.font.name = 'Consolas'
                    run.font.size = Pt(font_size - 1)
                else:
                    # Check for inline math $math$
                    math_parts = re.split(r'(\$.*?\$)', sp)
                    for mp in math_parts:
                        if mp.startswith('$') and mp.endswith('$') and len(mp) > 2:
                            math_clean = clean_latex_to_unicode(mp[1:-1]).replace('___DOLLARSIGN___', '$')
                            m_run = paragraph.add_run(math_clean)
                            m_run.italic = True
                            m_run.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
                        else:
                            run = paragraph.add_run(mp.replace('___DOLLARSIGN___', '$'))
                            run.bold = bold
                            run.italic = italic
                            if color:
                                run.font.color.rgb = color
                            if font_size:
                                run.font.size = Pt(font_size)


def add_equation_box(doc, eq_text):
    """Creates an elegant shaded callout box for mathematical equations in Word."""
    clean_eq = clean_latex_to_unicode(eq_text)
    
    # 1-cell callout table
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    
    # Light blue-gray background
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=100, bottom=100, left=180, right=180)
    
    # Add blue left border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="1E88E5"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    
    run = p.add_run(clean_eq)
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x0A, 0x25, 0x40)
    
    # Spacer
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(2)
    p_space.paragraph_format.space_after = Pt(4)


def build_docx():
    print(f"Reading {REPORT_MD}...")
    with open(REPORT_MD, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = Document()

    # Set 1-inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base font setup
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    in_table = False
    table_lines = []

    def flush_table(tbl_lines):
        if not tbl_lines:
            return
        rows_data = []
        for l in tbl_lines:
            if re.match(r'^\s*\|?\s*[-:]+[-| :]*$', l):
                continue
            cells = [c.strip() for c in l.strip().strip('|').split('|')]
            if any(cells):
                rows_data.append(cells)
                
        if not rows_data:
            return

        cols_count = max(len(r) for r in rows_data)
        table = doc.add_table(rows=len(rows_data), cols=cols_count)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        for r_idx, row in enumerate(rows_data):
            for c_idx in range(cols_count):
                cell_text = row[c_idx] if c_idx < len(row) else ""
                cell = table.cell(r_idx, c_idx)
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                
                is_header = (r_idx == 0)
                add_formatted_text(p, cell_text, bold=is_header, font_size=9.5)
                
                if is_header:
                    set_cell_background(cell, "0D47A1")
                    for run in p.runs:
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        run.bold = True
                elif r_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")
                    
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check if line is part of a markdown table
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            flush_table(table_lines)
            table_lines = []
            in_table = False

        # Equation display block ($$ ... $$)
        if line.strip().startswith('$$'):
            eq_lines = []
            if line.strip().endswith('$$') and len(line.strip()) > 2:
                # Single-line display equation
                eq_text = line.strip()[2:-2].strip()
                add_equation_box(doc, eq_text)
                i += 1
                continue
            else:
                # Multi-line display equation
                line_content = line.strip()[2:].strip()
                if line_content:
                    eq_lines.append(line_content)
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    eq_lines.append(lines[i].rstrip())
                    i += 1
                if i < len(lines):
                    last_l = lines[i].strip()[:-2].strip()
                    if last_l:
                        eq_lines.append(last_l)
                eq_text = "\n".join(eq_lines)
                add_equation_box(doc, eq_text)
                i += 1
                continue

        # Headings
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(line[2:])
            run.font.name = 'Georgia'
            run.font.size = Pt(20)
            run.bold = True
            run.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)
        elif line.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(line[3:])
            run.font.name = 'Georgia'
            run.font.size = Pt(14)
            run.bold = True
            run.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)
        elif line.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(line[4:])
            run.font.name = 'Georgia'
            run.font.size = Pt(12)
            run.bold = True
            run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        elif line.startswith('#### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line[5:])
            run.font.size = Pt(11)
            run.bold = True
        elif line.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            add_formatted_text(p, line[2:], italic=True, color=RGBColor(0x1E, 0x88, 0xE5))
        elif line.startswith('- ') or line.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_text(p, line[2:])
        elif re.match(r'^\d+\.\s', line):
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            item_text = re.sub(r'^\d+\.\s', '', line)
            add_formatted_text(p, item_text)
        elif line.startswith('---'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
        elif line.strip() == "":
            pass
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i].rstrip())
                i += 1
            if code_lines and not any('graph' in cl for cl in code_lines):
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                tbl.autofit = False
                cell = tbl.cell(0, 0)
                cell.width = Inches(6.5)
                set_cell_background(cell, "F8FAFC")
                set_cell_margins(cell, top=60, bottom=60, left=140, right=140)
                tcPr = cell._tc.get_or_add_tcPr()
                borders = parse_xml(f'''
                    <w:tcBorders {nsdecls("w")}>
                        <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                        <w:left w:val="single" w:sz="18" w:space="0" w:color="475569"/>
                        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                        <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                    </w:tcBorders>
                ''')
                tcPr.append(borders)
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                code_text = "\n".join(code_lines)
                run = p.add_run(code_text)
                run.font.name = 'Consolas'
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                p_space = doc.add_paragraph()
                p_space.paragraph_format.space_before = Pt(1)
                p_space.paragraph_format.space_after = Pt(2)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text(p, line)
            
        i += 1

    if in_table and table_lines:
        flush_table(table_lines)

    # Append high-resolution figures section
    doc.add_page_break()
    p_fig_head = doc.add_paragraph()
    run = p_fig_head.add_run("Appendix B: Empirical Evaluation Figures")
    run.font.name = 'Georgia'
    run.font.size = Pt(16)
    run.bold = True
    run.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    figures = [
        ("figure_1_pr_curve_comparison.png", "Figure 1: Precision-Recall Curve Comparison (Baseline vs Proposed BCBSMOTE)."),
        ("figure_2_confusion_matrices.png", "Figure 2: Confusion Matrix Side-by-Side Comparison on European Cardholders Test Set."),
        ("figure_3_cost_sensitive_curve.png", "Figure 3: Cost-Sensitive Expected Financial Loss vs Classification Threshold."),
        ("figure_4_ablation_study.png", "Figure 4: Four-Stage Systematic Ablation Study Performance Comparison."),
        ("figure_5_classifier_benchmark.png", "Figure 5: Alternative Model Architecture Performance Benchmarking."),
        ("figure_6_noise_stress_curve.png", "Figure 6: Robustness Degradation under Label Noise Injection (0% to 10%).")
    ]

    for fname, caption in figures:
        fpath = os.path.join(ASSET_DIR, fname)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(12)
            p_img.paragraph_format.space_after = Pt(4)
            doc.add_picture(fpath, width=Inches(5.8))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(14)
            c_run = p_cap.add_run(caption)
            c_run.italic = True
            c_run.font.size = Pt(9.5)
            c_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    saved_targets = []
    for target in [REPORT_DOCX, "reports/technical_review_report_fixed.docx", "reports/technical_review_report_updated.docx"]:
        try:
            doc.save(target)
            saved_targets.append(target)
            print(f"Successfully generated clean Word document at: {target}")
        except PermissionError:
            print(f"Notice: '{target}' is currently locked by Word.")
    
    if not saved_targets:
        emergency_docx = "reports/technical_review_report_latest.docx"
        doc.save(emergency_docx)
        print(f"Saved latest document to: {emergency_docx}")


if __name__ == '__main__':
    build_docx()
