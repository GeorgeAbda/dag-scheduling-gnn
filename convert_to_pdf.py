#!/usr/bin/env python3
"""Convert markdown to PDF using fpdf2 with Unicode support."""

import subprocess
import sys
import os

def check_and_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

def sanitize_text(text):
    """Replace unicode characters with ASCII equivalents."""
    replacements = {
        '\u2014': '--',  # em dash
        '\u2013': '-',   # en dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',   # non-breaking space
        '\u2022': '*',   # bullet
        '\u03a6': 'Phi', # Greek Phi
        '\u03c6': 'phi', # Greek phi
        '\u2192': '->',  # right arrow
        '\u2190': '<-',  # left arrow
        '\u2248': '~',   # approximately equal
        '\u2260': '!=',  # not equal
        '\u2264': '<=',  # less than or equal
        '\u2265': '>=',  # greater than or equal
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text

def create_pdf_with_fpdf():
    """Create PDF using fpdf2 library."""
    check_and_install("fpdf2")
    
    from fpdf import FPDF
    import re
    
    # Read the markdown file
    with open("research_report_platonic_biology.md", "r") as f:
        md_content = f.read()
    
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_page()
            self.set_auto_page_break(auto=True, margin=15)
            
        def chapter_title(self, title, level=1):
            title = sanitize_text(title)
            if level == 1:
                self.set_font("Helvetica", "B", 16)
                self.ln(10)
            elif level == 2:
                self.set_font("Helvetica", "B", 14)
                self.ln(8)
            elif level == 3:
                self.set_font("Helvetica", "B", 12)
                self.ln(6)
            else:
                self.set_font("Helvetica", "B", 11)
                self.ln(4)
            self.multi_cell(0, 8, title)
            self.ln(4)
            
        def body_text(self, text):
            text = sanitize_text(text)
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 6, text)
            self.ln(2)
            
        def quote_text(self, text):
            text = sanitize_text(text)
            self.set_font("Helvetica", "I", 10)
            self.set_x(20)
            self.multi_cell(170, 6, text)
            self.set_x(10)
            self.ln(4)
            
        def bullet_point(self, text):
            text = sanitize_text(text)
            self.set_font("Helvetica", "", 10)
            self.set_x(15)
            self.multi_cell(175, 6, f"* {text}")
            self.ln(1)
    
    pdf = PDF()
    pdf.set_margins(15, 15, 15)
    
    lines = md_content.split('\n')
    i = 0
    in_list = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Handle headers
        if line.startswith('# '):
            pdf.chapter_title(line[2:], 1)
        elif line.startswith('## '):
            pdf.chapter_title(line[3:], 2)
        elif line.startswith('### '):
            pdf.chapter_title(line[4:], 3)
        elif line.startswith('#### '):
            pdf.chapter_title(line[5:], 4)
        # Handle horizontal rules
        elif line.startswith('---'):
            pdf.ln(5)
        # Handle blockquotes
        elif line.startswith('>'):
            quote = line[1:].strip()
            pdf.quote_text(quote)
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:]
            # Remove markdown formatting
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            pdf.bullet_point(text)
        # Handle numbered lists
        elif re.match(r'^\d+\.', line):
            text = re.sub(r'^\d+\.\s*', '', line)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            pdf.bullet_point(text)
        # Regular paragraph
        else:
            # Clean up markdown formatting
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            
            # Collect full paragraph
            paragraph = text
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith(('#', '-', '*', '>', '`')):
                i += 1
                next_line = lines[i].strip()
                next_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', next_line)
                next_line = re.sub(r'\*([^*]+)\*', r'\1', next_line)
                next_line = re.sub(r'`([^`]+)`', r'\1', next_line)
                paragraph += ' ' + next_line
            
            pdf.body_text(paragraph)
        
        i += 1
    
    pdf.output("research_report_platonic_biology.pdf")
    print("PDF created successfully with fpdf2!")
    return True

if __name__ == "__main__":
    print("Creating PDF from markdown...")
    
    # Try fpdf2 approach (most reliable, pure Python)
    if create_pdf_with_fpdf():
        print("\nPDF saved as: research_report_platonic_biology.pdf")
    else:
        print("Failed to create PDF")
        sys.exit(1)
