# GenAI Interview Notes (RAG + Web Search/Tavily)

Based on the two things actually implemented in this repo: **RAG** (PDF → Chunks → MiniLM Embeddings → Chroma → Sarvam) and **Web Search** (Tavily → context → Sarvam).

This is written to be spoken naturally in an interview (not textbook-only definitions).

---

## PART 1 — RAG Interview Q&A

### Q1. Tell me about the RAG pipeline you implemented.

**Answer (speak naturally):**

I implemented a Python-based RAG pipeline where I use a PDF as the knowledge source. First, I extract the text using `pypdf` and split the extracted text into **1000-character chunks**. I generate embeddings for those chunks using **`all-MiniLM-L6-v2`** and store both the original chunks and their embeddings in **Chroma**.

When a user asks a document-related question, I generate an embedding for the question using the same embedding model and perform **similarity search in Chroma**. I retrieve the top five closest chunks and check the **best result against a distance threshold of 1.4**. If the document retrieval is relevant, I pass the retrieved text along with the question to **Sarvam 105B** to generate the answer.

If the document retrieval is weak, my application falls back to **Tavily Web Search**.

### Q2. What is RAG and why did you use it?

RAG means Retrieval-Augmented Generation. I used it because the LLM itself doesn’t have access to my private project documentation. Instead of expecting the model to already know my PDF, I retrieve relevant chunks from my document and provide them as context to the LLM. That makes the answers grounded in my data.

### Q3. Why didn’t you just send the entire PDF to the LLM?

Sending the entire PDF every time would be inefficient and can hurt answer quality—more irrelevant text, higher latency, and more tokens. With RAG, I only send the small chunks most relevant to the question.

### Q4. Explain your chunking strategy.

Right now I use simple character-based chunking: **split extracted PDF text into 1000-character chunks**. This makes retrieval manageable. A limitation is that I’m not using overlap, so a sentence can sometimes be split across chunks. In a production version, I would use recursive chunking with overlap and add metadata like page numbers.

### Q5. Why 1000 characters?

It’s a configuration choice for this app. I chose it to keep chunks small enough for focused retrieval but large enough to preserve useful context. It’s not “the best value” universally. I would evaluate chunk size and overlap for better retrieval.

### Q6. What is an embedding?

An embedding is a numerical vector that represents the semantic meaning of text. In my project, I use `all-MiniLM-L6-v2` to convert both document chunks and user questions into vectors, so similarity search can find meaningfully related chunks.

### Q7. Why did you use MiniLM?

MiniLM (`all-MiniLM-L6-v2`) is lightweight and works well for semantic similarity and retrieval. In this implementation it produces **384-dimensional embeddings** and runs locally via `sentence-transformers`.

### Q8. Is MiniLM the only embedding model?

No. It’s just what I picked. Other embedding options exist depending on retrieval quality, latency, cost, and deployment constraints.

### Q9. Why do you use the same embedding model for documents and questions?

Because the vectors must live in the same embedding space. I embed chunks during ingestion and embed the question during retrieval using the **same model**, so the similarity comparison is meaningful.

### Q10. What is the difference between an embedding and a vector?

In practice, embedding and vector are the same idea: the model outputs a vector representation. The embedding is the model-generated representation; the vector is the numeric structure you store and compare.

### Q11. Why store both embedding and original text in Chroma?

Chroma needs the embeddings to do similarity search. But the LLM needs the actual text chunks as context to generate the final answer. So we store both.

### Q12. Why did you choose Chroma?

Chroma is a simple local vector store. I used the **persistent** Chroma setup so the vectors are stored on disk and available between runs.

### Q13. Is Chroma the only vector DB?

No. Alternatives exist like Qdrant, Pinecone, Weaviate, Milvus, etc. Chroma was a good fit here because it’s straightforward for a local project.

### Q14. What happens when the user asks a question?

First, the router classifies the question into a mode. For document questions, I generate an embedding for the user question, query Chroma for the **top five** closest chunks, and use the best distance to decide if retrieval is strong enough. If yes, I pass the retrieved text context to Sarvam to answer.

### Q15. Does Chroma understand the meaning of the question?

Not in the LLM sense. Chroma doesn’t “understand language” like a model. It compares vectors. The embedding model creates semantic vectors, and Chroma does numeric similarity search.

### Q16. What similarity search are you doing?

Vector similarity search: encode question and compare it to stored chunk vectors. Then retrieve the closest chunks.

### Q17. Why top 5?

It’s a tradeoff. I want enough context for the LLM but not too many irrelevant chunks. It’s tunable; you’d adjust based on evaluation.

### Q18. What does distance mean?

Distance measures how far the question vector is from a chunk vector in the embedding space. In this implementation, **smaller distance means more similar**.

### Q19. Why did you use a 1.4 threshold?

It’s not universal. It’s specific to my embedding model + Chroma setup + document content. The right threshold depends on evaluation.

### Q20. What happens if the distance is 0.8?

0.8 is inside the 1.4 threshold, so the app uses the retrieved document chunks.

### Q21. What happens if the distance is 1.8?

That’s above the threshold, so I treat the document retrieval as weak and trigger the web-search fallback.

### Q22. What happens if only one of the five chunks is relevant?

In this implementation, the relevance gate uses the **best result’s distance** rather than requiring all five chunks to pass. So if the closest chunk is good, we proceed.

### Q23. What happens if the answer isn’t in the document?

If the retrieved chunks are weak, we fall back to Tavily web search. Even if retrieval is considered relevant, Sarvam is prompted to answer using only the provided context. If the context truly doesn’t contain the answer, it responds with the “I don’t know based on the provided context” rule.

### Q24. What are the limitations of your current RAG?

The biggest ones are: chunking is simple **character slicing** without overlap, I don’t use page metadata in retrieval, and there’s no reranking or hybrid retrieval. Also, the threshold is manually configured.

### Q25. How would you improve your current RAG?

I would improve chunking using recursive splitting with overlap and add metadata like page number. Then I’d retrieve more candidates and add a reranker to select the best chunks. I would also evaluate hybrid retrieval and build a small evaluation harness to tune chunking, top-k, and the relevance threshold.

---

## PART 2 — Web Search / Tavily Interview Q&A

### Q26. Why did you add Web Search to your RAG system?

My document knowledge base is limited to the PDF. So it can’t answer questions that require current information or public internet knowledge. I added Tavily web search so that when document retrieval is weak, the app can use live web context.

### Q27. What is Tavily?

Tavily is the web-search provider used by this project. It returns search results (and sometimes a short summary) that I then pass as context to Sarvam.

### Q28. Why Tavily instead of Sarvam?

Tavily is specifically for retrieval from the web. Sarvam is the LLM used for generating the final answer. I separate retrieval and generation.

### Q29. Is Tavily an LLM?

No. Tavily is a search/retrieval layer. Sarvam generates the final response.

### Q30. How does your Web Search flow work?

When the router selects WEB mode (for news/current public facts) or when DOC retrieval is weak, `web_search.py` calls Tavily. I build a context string from Tavily results—title, URL, and content snippet—and then use `generate_answer(question, context)` to produce the final answer.

### Q31. Why limit it to five results?

It’s a balance between coverage and noise. More results increases token usage and can add irrelevant context, so I keep it at five for now.

### Q32. What does `search_depth` do?

It controls how deep Tavily searches. In this repo I use `search_depth="basic"` for simplicity and speed.

### Q33. What information do you get from Tavily?

I get result titles, URLs, and content snippets, and sometimes a Tavily-provided short summary (called `answer` in the response).

### Q34. How do you prevent the LLM from searching the web itself?

The code explicitly calls Tavily first. Sarvam only receives the context text produced by Tavily or by PDF retrieval. Sarvam is not allowed to browse.

### Q35. What happens if Tavily fails?

In this project, web search is expected to be supported only when `TAVILY_API_KEY` exists. If it’s missing, the app raises an error. In a production system I’d add retries, timeouts, and safer fallback behavior.

### Q36. RAG vs Web Search?

RAG retrieves from my controlled knowledge base (the PDF). Web search retrieves from external public sources. RAG is best for private or stable knowledge; web is best for current/public facts.

### Q37. Can you combine RAG and Web Search?

Yes. In this app, DOC mode retrieves from Chroma and checks relevance. If the match is weak, it can switch to WEB mode and call Tavily.

### Q38. Why not always use Web Search instead of RAG?

Because my private document might not exist online. RAG ensures answers come from the controlled document knowledge base.

### Q39. Why not always use RAG instead of Web Search?

Because the PDF doesn’t contain current news or constantly changing public information. Web search complements RAG for those cases.

### Q40. How do you make sure web results are trustworthy?

In this repo, I don’t have a dedicated “source quality” or “citation verification” layer. I pass Tavily’s snippets into Sarvam as context. In production, I would add domain filters, reranking, and evaluation checks.

---

## PART 3 — Combined Architecture Questions

### Q41. Explain your DOC → WEB fallback.

When the router chooses DOC, I query Chroma for the top five chunks and look at the best distance. If the best distance is within my threshold (<= 1.4), I use document context. If it’s above 1.4, I treat document retrieval as weak and call Tavily to build a web context. Then I generate the answer using that context.

### Q42. What is the role of the router?

The router decides which capability should handle the question. This prevents every query from going through both retrieval systems. It also enables greeting/chat/time handling.

### Q43. Why do you need a router if you already have RAG?

Because RAG is only one capability. Greetings don’t need retrieval, and web/public questions shouldn’t depend only on the PDF.

### Q44. Explain the difference between your LLM and embedding model.

The embedding model turns text into vectors for retrieval. The LLM (Sarvam) uses the retrieved context and generates the natural language answer. They have different responsibilities.

### Q45. Can you replace Sarvam with OpenAI?

Yes. The architecture can keep Chroma retrieval the same conceptually; you’d just replace the generation call with OpenAI’s chat API.

### Q46. Can you replace MiniLM with OpenAI embeddings?

Yes, but you must rebuild the vector store because the embedding space changes. You can’t mix vectors from different models.

### Q47. Can you replace Chroma?

Yes. Chroma is the vector store implementation. Conceptually you can use another vector DB as long as you support embeddings + similarity search + persistence.

### Q48. What is the biggest weakness in your current RAG?

Simple chunking without overlap and no reranking/hybrid retrieval, plus manual thresholding. The biggest improvement areas are retrieval quality and evaluation-driven tuning.

---

## PART 4 — Scenario Questions

### Q49. What if the PDF has 500 pages?

I would still not send the full PDF to the LLM. I would extract text, chunk it, embed it, and store it in the vector DB. At query time I would retrieve only the most relevant chunks.

### Q50. What if the answer is split between two chunks?

With no overlap chunking, that can happen. I would improve chunking with overlap or recursive chunking so that semantic boundaries are preserved.

### Q51. What if the user asks using different words than the PDF?

That’s exactly where semantic embeddings help. Retrieval is based on vector similarity, so it doesn’t require the question to match exact keywords in the PDF.

### Q52. What if Chroma returns irrelevant chunks?

That’s why I have a relevance threshold and fallback to WEB when document retrieval looks weak. In a stronger production system, I’d add reranking and more robust evaluation.

### Q53. What if the web search returns irrelevant results?

In this repo, I don’t have a deep source-quality layer. A production system would filter or rerank sources and ensure the retrieved snippets actually support the answer.

### Q54. What if the PDF changes?

I would reprocess the PDF—regenerate chunks and embeddings—and update the vector DB. If the embedding model changes, rebuild the store completely.

### Q55. What if you change the embedding model?

I would regenerate embeddings and rebuild the vector DB. I wouldn’t mix vectors from old and new embedding models because they’re in different vector spaces.

---

## PART 5 — The 10 Questions to Prepare (Highly Recommended)

1. Explain your complete RAG pipeline.
2. Why did you use embeddings?
3. Why MiniLM?
4. Why do you use the same embedding model for documents and queries?
5. Why Chroma? What alternatives exist?
6. Explain similarity search and distance.
7. Why top 5 and why threshold 1.4?
8. Why Tavily? RAG vs Web Search?
9. Explain your DOC → WEB fallback.
10. How would you improve your current RAG?

