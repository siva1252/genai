1. Transformer
→ Sentence lo tokens madhya contextual relationships ni process chestundi.

2. Enduku relationship kavali?
→ Oka word meaning/context ni correct ga understand cheyyadaniki, surrounding words important.

3. Attention
→ Transformer lo unna core mechanism.
→ Current token ki other tokens entha important/relevant ani calculate chestundi.


------------process-------------------

RAG
 ↓
Relevant chunk TEXT
 ↓
Question + Chunk TEXT
 ↓
Tokenization
 ↓
Tokens
 ↓
Token IDs
 ↓
Embedding vectors
 ↓
Transformer
 ↓
Attention + other processing
 ↓
Output tokens
 ↓
Text answer


----------------------main concepts -------------------------------------


Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding Vectors
 ↓
Q / K / V
 ↓
Attention Scores
 ↓
Softmax → Attention Weights
 ↓
Weighted V
 ↓
Contextual Representation
 ↓
Residual + Normalization
 ↓
FFN
 ↓
Residual + Normalization
 ↓
Next Transformer Block
 ↓
...
 ↓
Final Hidden Representation
 ↓
LM Head
 ↓
Logits
 ↓
Probabilities
 ↓
Next Token ID
 ↓
Repeat
 ↓
Decode
 ↓
Final Text