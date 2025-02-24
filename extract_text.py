import os
import fitz  # PyMuPDF

def extract_text_from_pdfs(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in doc])

            output_file = os.path.join(output_folder, pdf_file.replace(".pdf", ".txt"))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted: {pdf_file}")

# Run the function
extract_text_from_pdfs("pdfs", "texts")

