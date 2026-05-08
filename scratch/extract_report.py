import pypdf
import os

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

target_file = "reference/차세대 스파이킹 트랜스포머(Spiking Transformer)를 위한 동적 연산 에너지(Dynamic Compute Energy) 측정 모델 및 Extended-Adaptive MBE 뉴런 기반의 효율성 최적화 심층 분석.pdf"
output_file = "reference/report_clean.txt"

if os.path.exists(target_file):
    extract_pdf_to_txt(target_file, output_file)
else:
    print(f"File not found: {target_file}")
    # Try with wildcard if direct name fails
    import glob
    files = glob.glob("reference/*동적*.pdf")
    if files:
        extract_pdf_to_txt(files[0], output_file)
    else:
        print("No matching files found with wildcard.")
