from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import json

def generate_pdf_report(report_data: dict, output_path: str):
    """
    Generate a PDF report based on provided report data.

    report_data should be a dict such as:
    {
       "business_overview": "string",
       "key_findings": "string",
       "extracted_data": { ... },
       "calculated_ratios": { ... }
    }
    """
    c = canvas.Canvas(output_path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75 * inch
    line_height = 14

    def draw_section(title, text, start_y):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, start_y, title)
        start_y -= 18
        c.setFont("Helvetica", 10)
        # Split the text into lines that fit the page
        lines = text.splitlines() if isinstance(text, str) else json.dumps(text, indent=2).splitlines()
        for line in lines:
            if start_y <= margin:
                c.showPage()
                start_y = height - margin
            c.drawString(margin, start_y, line)
            start_y -= line_height
        return start_y - 10

    y = height - margin
    title = "Financial Analysis Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, y, title)
    y -= 30

    sections = [
        ("Business Overview", report_data.get("business_overview", "")),
        ("Key Findings", report_data.get("key_findings", "")),
        ("Extracted Data", report_data.get("extracted_data", {})),
        ("Calculated Ratios", report_data.get("calculated_ratios", {}))
    ]

    for section_title, content in sections:
        y = draw_section(section_title, content, y)
        if y <= margin:
            c.showPage()
            y = height - margin

    c.save()