from pypdf import PdfReader

PDF_PATH = "uploads/PMS_Project_Documentation.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_chunks(pdf_path=PDF_PATH, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break

    return chunks, text, len(reader.pages)


if __name__ == "__main__":
    chunks, text, page_count = get_chunks()

    print(f"Total Pages: {page_count}")
    print(f"Total Characters: {len(text)}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Overlap: {CHUNK_OVERLAP}")

    if chunks:
        print("\n--- CHUNK BOUNDARIES (with overlap) ---")
        for i in range(min(8, len(chunks) - 1)):
            print(f"\n=== end of chunk {i} ===")
            print(chunks[i][-120:])
            print(f"=== start of chunk {i+1} ===")
            print(chunks[i+1][:120])