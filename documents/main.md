   ├── Chunk Overlap       ← START HERE   
  this means we can edges of chunk we can add another add chunk beacuase we don't miss the any data so we use chunk overlap.

  
Metadata  - this means we save every chunk data like pages and everything and that chunk id and pages id also liek this we use metadata

it's usful of multiple pdfs also so that time metadata wil be helpfull perfectlyy
main usess
1. Source tracking
2. Filtering

| Metric            | What it measures         | Better             |
| ----------------- | ------------------------ | ------------------ |
| L2 Distance       | How far vectors are      | **Lower**          |
| Cosine Similarity | Directional similarity   | **Higher**         |
| Cosine Distance   | Distance based on cosine | **Lower**          |
| Dot Product       | Vector alignment/product | Usually **Higher** |


User Question
      ↓
    MiniLM
      ↓
Question Vector
      ↓
    Chroma
      ↓
Compare with Chunk Vectors
      ↓
Calculate L2 Distances
      ↓
Rank results
      ↓
Top-K = 5
      ↓
Best/lowest distance
      ↓
Compare with 1.4
      ↓
 ┌────┴────┐
 ↓         ↓
≤ 1.4    > 1.4
 ↓         ↓
PDF       Web
 ↓         ↓
 └────┬────┘
      ↓
    Sarvam
      ↓
   Answer

   ├── Top-K
based in question it chnages to question vector so it find close top 5 like we use k=5 take best distance so based on that we have thershold to compare is smaller it checks okay if it greater it goes to tavily this is the thing crct 

so we can check every pdf questions to find chunks distance and after that we can check without pdf questions aldo and find there chukns to comapre know find one threshold so we can find like that

Reranking means 
based on the question we can use top k chormo returns top chunks so based on lowest score distance it compares thresholds  this is noraml rag

if reranking measn same we can take top k after that based on question it give top chormo retunrs top chunks so based on that again reranking will be checks question + chunks it gives another revalnce score know it gives again top 2 chunks so based on thta chunk it gves direct to llm so thsi is rernaking


Hybrid search means 

User Question
      ↓
 ┌───────────────┬───────────────┐
 ↓                               ↓
Semantic Search              Keyword Search
(MiniLM + Chroma)              (BM25)
 ↓                               ↓
Meaning-based results        Exact-term results
 └───────────────┬───────────────┘
                 ↓
          Combine rankings/scores
                 ↓
          Best relevant chunks
                 ↓
               LLM
                 ↓
             Answer

it combines of both semantic search  + keyowrd search 
semantic it gives perfect meaningfully answers 
keyword it gives perfect keyword like api names and keywords
better retrieval gives the LLM better context, which can improve answer quality."


Rag evaluation 


                    USER QUESTION
                          ↓
                     RETRIEVAL
                          ↓
                   Relevant Context
                          ↓
                         LLM
                          ↓
                        Answer
                          ↓
             ┌────────────┴────────────┐
             ↓                         ↓
        RAG EVALUATION              GROUNDING
             ↓                         ↓
      "Was retrieval/answer      "Is answer supported
          good?"                    by context?"
             │                         │
             └────────────┬────────────┘
                          ↓
                    HALLUCINATION
                          ↓
                "Did model invent
                 unsupported info?"


                 
RAG Evaluation

"I create representative questions with expected answers/sources and evaluate retrieval and generation separately. I can use that dataset to tune Top-K, thresholds, chunking, reranking and other retrieval settings."


Grounding

"I verify that the generated answer is supported by the retrieved context and doesn't introduce unsupported claims."

Hallucination

"Hallucination is when the LLM produces unsupported or fabricated information. RAG, better retrieval, grounded prompts and evaluation can reduce the risk, but they don't guarantee zero hallucinations."

RAG Evaluation
→ checks retrieval + context + answer quality

Grounding
→ ❌ Answer contains a claim not supported by context

Hallucination
→ ✅ "5-node cluster" is an unsupported generated claim