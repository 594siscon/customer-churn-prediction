import markdown
import subprocess
import os

def main():
    md_file = 'report.md'
    html_file = 'report.html'
    pdf_file = 'report.pdf'
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
        return
        
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
        
    title = "電商客戶價值分析與預測報告"
    subtitle = "基於隨機森林與 TreeSHAP 歸因的深度分析"
    
    # Convert markdown to html with extra extensions, including toc for dynamic table of contents.
    html_body = markdown.markdown(
        md_content,
        extensions=['extra', 'toc'],
        extension_configs={
            'toc': {
                'title': '目錄',
                'toc_depth': '2-3'
            }
        }
    )
    
    # Let's wrap it in a beautiful CSS layout.
    css_styles = """
    @page {
        size: A4;
        margin: 2.5cm 2cm 2.5cm 2cm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        font-size: 11pt;
        line-height: 1.7;
        color: #2d3748;
        background-color: #ffffff;
    }
    
    /* Cover Page Styling */
    .cover-page {
        page-break-after: always;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        padding-top: 5cm;
        box-sizing: border-box;
    }
    .cover-title {
        font-size: 28pt;
        font-weight: 800;
        color: #1a365d;
        line-height: 1.3;
        margin-bottom: 15px;
        border-bottom: 4px solid #3182ce;
        padding-bottom: 20px;
        width: 100%;
    }
    .cover-subtitle {
        font-size: 16pt;
        color: #4a5568;
        margin-bottom: 5cm;
    }
    .cover-meta {
        font-size: 11pt;
        color: #718096;
        line-height: 1.8;
        margin-top: auto;
    }
    
    /* Content Styling */
    h1 {
        display: none; /* Hide the main H1 since it's on the cover page */
    }
    h2 {
        font-size: 16pt;
        color: #1a365d;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 35px;
        margin-bottom: 15px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 13pt;
        color: #2b6cb0;
        margin-top: 25px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }
    h4 {
        font-size: 11pt;
        color: #2d3748;
        margin-top: 20px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }
    p {
        margin-bottom: 16px;
        text-align: justify;
    }
    ul, ol {
        margin-bottom: 16px;
        padding-left: 20px;
    }
    li {
        margin-bottom: 6px;
    }
    
    /* Table Styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 10pt;
        break-inside: avoid;
    }
    th, td {
        border: 1px solid #e2e8f0;
        padding: 10px 12px;
        text-align: left;
    }
    th {
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 700;
    }
    tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    /* Image Styling */
    img {
        max-width: 90%;
        height: auto;
        display: block;
        margin: 25px auto;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        break-inside: avoid;
    }
    
    /* Alert Block (Blockquote) Styling */
    blockquote {
        margin: 20px 0;
        padding: 15px 20px;
        background-color: #f7fafc;
        border-left: 4px solid #cbd5e0;
        color: #4a5568;
        border-radius: 0 4px 4px 0;
        break-inside: avoid;
    }
    
    blockquote p {
        margin: 0;
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 40px 0;
    }
    
    /* Page Break Helper */
    .page-break {
        page-break-after: always;
    }
    
    /* Table of Contents (TOC) Styling */
    .toc {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px 28px;
        margin: 30px 0;
        break-inside: avoid;
    }
    .toctitle {
        display: block;
        font-size: 14pt;
        font-weight: 700;
        color: #1a365d;
        margin-top: 0;
        margin-bottom: 15px;
        border-bottom: 2px solid #cbd5e0;
        padding-bottom: 8px;
    }
    .toc ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
    }
    .toc li {
        margin-bottom: 10px;
        font-size: 11pt;
        font-weight: 600;
    }
    .toc li a {
        color: #2b6cb0;
        text-decoration: none;
    }
    .toc li a:hover {
        color: #2c5282;
        text-decoration: underline;
    }
    .toc ul ul {
        padding-left: 20px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .toc ul ul li {
        font-size: 10pt;
        font-weight: 400;
        margin-bottom: 6px;
    }
    .toc ul ul li a {
        color: #4a5568;
    }
    """
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="cover-page">
        <div class="cover-title">{title}</div>
        <div class="cover-subtitle">{subtitle}</div>
        <div class="cover-meta">
            <strong>報告類型：</strong> 數據分析與機器學習模型歸因報告<br>
            <strong>使用模型：</strong> 隨機森林分類器 (Random Forest Classifier)<br>
            <strong>歸因技術：</strong> TreeSHAP (SHAP 互動值與全局分析)<br>
            <strong>製作時間：</strong> 2026年6月
        </div>
    </div>
    
    <div class="content-body">
        {html_body}
    </div>
</body>
</html>
"""

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    print(f"Generated HTML at {html_file}")
    
    # Run Edge headless PDF export
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(html_file)
    abs_pdf = os.path.abspath(pdf_file)
    
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        abs_html
    ]
    
    print("Running Edge to convert HTML to PDF...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Success! PDF generated at:", abs_pdf)
    except subprocess.CalledProcessError as e:
        print("Error during PDF generation:")
        print(e.stderr)
        print(e.stdout)

if __name__ == "__main__":
    main()
