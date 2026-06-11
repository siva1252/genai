from pypdf import PdfReader

reader = PdfReader("uploads/PMS_Project_Documentation.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(f"Total Pages: {len(reader.pages)}")
print(f"Total Characters: {len(text)}")

print(text[:1000])


chunks=[]
chunk_size=1000
for i in range(0, len(text), chunk_size):
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)
print(f"Total Chunks: {len(chunks)}")
print(f"First Chunk: {chunks[0]}")
print(f"Last Chunk: {chunks[1]}")


