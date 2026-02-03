import fitz
import io

def extract_pdf_content(file_input):
    """
    Extracts text from a PDF file input (bytes, stream, or path).
    """
    doc = None
    if isinstance(file_input, bytes):
        doc = fitz.open(stream=file_input, filetype="pdf")
    elif hasattr(file_input, "read"):
        file_bytes = file_input.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        # Assume path
        doc = fitz.open(file_input)

    sections = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        page_text = []
        for b in blocks:
            if isinstance(b[4], str):
                page_text.append(b[4].strip())

        full_text = "\n".join(page_text)
        
        # 📊 Table Extraction (Low-code Multi-modal)
        # We append structured table data to help the LLM understand rows/cols
        try:
            tables = page.find_tables()
            if tables:
                for i, table in enumerate(tables):
                    # headers=None to auto-detect or just standard 2D extraction
                    # extract() returns a list of lists [row][col]
                    table_data = table.extract() 
                    if table_data:
                        # Simple markdown conversion
                        md_table = "\n\n**[Table Extracted]**\n"
                        # Header
                        if len(table_data) > 0:
                            md_table += "| " + " | ".join([str(c) for c in table_data[0]]) + " |\n"
                            md_table += "| " + " | ".join(["---"] * len(table_data[0])) + " |\n"
                            # Rows
                            for row in table_data[1:]:
                                md_table += "| " + " | ".join([str(c).replace("\n", " ") for c in row]) + " |\n"
                        
                        full_text += md_table + "\n"
        except Exception:
            # Fallback if table extraction fails (e.g. old pymupdf version)
            pass

        lower_text = full_text.lower()

        section = "unknown"
        if "conclusion" in lower_text:
            section = "conclusion"
        elif "abstract" in lower_text:
            section = "abstract"
        elif "result" in lower_text:
            section = "results"
        elif "reference" in lower_text:
            section = "references"

        sections.append({
            "page": page_num + 1,
            "content": full_text,
            "section": section
        })

    return sections

