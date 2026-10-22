"""
Exports reports/technical_review_report.md to an academic-styled HTML document:
reports/technical_review_report.html
Designed for clean PDF export via Chrome/Edge/Firefox 'Print to PDF' (Ctrl + P).
"""

import os
import markdown

REPORT_MD = "reports/technical_review_report.md"
REPORT_HTML = "reports/technical_review_report.html"

with open(REPORT_MD, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert markdown
html_body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])

# Inject image embeds where relevant
asset_dir = "../results/report_assets"
image_insertions = f"""
<div style="margin: 40px 0; text-align: center;">
    <h2 style="text-align: left; border-bottom: 2px solid #0D47A1; padding-bottom: 8px;">Appendix B: Empirical Evaluation Figures</h2>
    <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 24px; margin-top: 20px;">
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_1_pr_curve_comparison.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Precision-Recall Comparison">
            <p><em>Figure 1: Precision-Recall Curve Comparison (Baseline vs Proposed BCBSMOTE)</em></p>
        </div>
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_2_confusion_matrices.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Confusion Matrix Comparison">
            <p><em>Figure 2: Confusion Matrix Side-by-Side Evaluation</em></p>
        </div>
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_3_cost_sensitive_curve.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Cost Sensitive Curve">
            <p><em>Figure 3: Cost-Sensitive Expected Financial Loss vs Threshold</em></p>
        </div>
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_4_ablation_study.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Ablation Study">
            <p><em>Figure 4: Systematic 4-Stage Ablation Study Performance</em></p>
        </div>
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_5_classifier_benchmark.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Classifier Benchmark">
            <p><em>Figure 5: Alternative Model Architecture Performance</em></p>
        </div>
        <div style="flex: 1 1 45%; min-width: 320px;">
            <img src="{asset_dir}/figure_6_noise_stress_curve.png" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Stress Testing Curve">
            <p><em>Figure 6: Robustness Degradation under Label Noise Injection (0%-10%)</em></p>
        </div>
    </div>
</div>
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Technical Review Report - Credit Card Fraud Detection</title>
<!-- MathJax for rendering LaTeX mathematical equations -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Open+Sans:wght@400;600;700&display=swap');
    
    body {{
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.7;
        color: #222;
        background-color: #fafafa;
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 60px 70px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-radius: 4px;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Merriweather', Georgia, serif;
        color: #111;
        font-weight: 700;
        line-height: 1.3;
    }}
    h1 {{
        font-size: 26px;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 12px;
        margin-top: 0;
    }}
    h2 {{
        font-size: 20px;
        margin-top: 36px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 8px;
        color: #0D47A1;
    }}
    h3 {{
        font-size: 16px;
        margin-top: 24px;
        color: #333;
    }}
    p, li {{
        font-size: 14.5px;
        text-align: justify;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 13.5px;
    }}
    th, td {{
        border: 1px solid #d0d7de;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    code {{
        font-family: Consolas, "Courier New", monospace;
        background-color: #f1f5f9;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 13px;
    }}
    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 14px;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 12.5px;
    }}
    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}
    blockquote {{
        border-left: 4px solid #1E88E5;
        margin: 18px 0;
        padding: 8px 18px;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-style: italic;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 30px 0;
    }}
    @media print {{
        body {{
            background: #fff;
            padding: 0;
        }}
        .container {{
            box-shadow: none;
            padding: 0;
            max-width: 100%;
        }}
        h2 {{
            page-break-before: always;
        }}
    }}
</style>
</head>
<body>
<div class="container">
{html_body}
<hr>
{image_insertions}
</div>
</body>
</html>
"""

with open(REPORT_HTML, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Successfully exported styled report to: {REPORT_HTML}")
