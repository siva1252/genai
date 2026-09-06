# OpenAI API Endpoints Cheat Sheet (Very Important)

| What you want to do | API you call |
| --- | --- |
| Ask a question / Generate text | `client.responses.create()` ✅ (Recommended) |
| Chatbot with System/User/Assistant messages | `client.chat.completions.create()` |
| Create embeddings (RAG) | `client.embeddings.create()` |
| Generate an image | `client.images.generate()` |
| Edit an existing image | `client.images.edit()` |
| Speech → Text (Transcription) | `client.audio.transcriptions.create()` |
| Text → Speech | `client.audio.speech.create()` |
