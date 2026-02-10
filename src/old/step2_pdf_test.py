from pypdf import PdfReader

pdf_path = r"D:\Requlatory-Chatbot\data\ich\Q9.pdf"

reader = PdfReader(pdf_path)
print(f"Number of pages: {len(reader.pages)}")

first_page = reader.pages[0]
text = first_page.extract_text()

print("\n--- First page text ---\n")

if text:
    print(text)
else:
    print("No extractable text found.")