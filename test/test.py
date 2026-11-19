
import os
from dotenv import load_dotenv
import cohere
from groq import Groq

load_dotenv()

# Test 1: Cohere Embeddings
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
embed_model = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
embedding_response = co.embed(
    texts=["Hello DocuMind Knowledge Assistant"],
    model=embed_model,
    input_type="search_document",
    embedding_types=["float"]
)

dimension = len(embedding_response.embeddings.float[0])
print("Embedding dimension:", dimension)

# Test 2: Groq Chat Completion
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chat_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
chat_response = groq_client.chat.completions.create(
    model=chat_model,
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ],
    max_tokens=200
)

print(
    "Groq Response:",
    chat_response.choices[0].message.content
)
