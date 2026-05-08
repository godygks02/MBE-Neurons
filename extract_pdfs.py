import pypdf

def extract_pdf_to_txt(pdf_path, txt_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))
        print(f"Successfully extracted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

extract_pdf_to_txt('MBE뉴런.pdf', 'MBE뉴런.txt')
extract_pdf_to_txt('Training-Free ANN-to-SNN Conversion for High-Performance.pdf', 'Paper.txt')
