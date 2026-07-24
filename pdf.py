from pypdf import PdfReader

PDF_PATH = "uploads/PMS_Project_Documentation.pdf"
CHUNK_SIZE = 1000


def get_chunks(pdf_path=PDF_PATH, chunk_size=CHUNK_SIZE):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks, text, len(reader.pages)


if __name__ == "__main__":
    chunks, text, page_count = get_chunks()

    print(f"Total Pages: {page_count}")
    print(f"Total Characters: {len(text)}")
    print(text[:1000])
    print(f"Total Chunks: {len(chunks)}")

    if chunks:
        print(f"First Chunk: {chunks[0]}")
        print(f"Last Chunk: {chunks[-1]}")
