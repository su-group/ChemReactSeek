import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Load local pre-trained embedding model
model = SentenceTransformer('./all-MiniLM-L6-v2')  #The storage location where you downloaded the all-MiniLM-L6-v2 model

# Initialize FAISS index
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Store document chunks and their corresponding IDs
documents = []


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()  # Remove leading/trailing whitespace
    return text


def split_into_chunks(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # Ensure chunk is not empty
            chunks.append(chunk)
    return chunks


def vectorize_and_store(chunks):
    if not chunks:
        print("No chunks to vectorize.")
        return

    embeddings = model.encode(chunks)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)  # If one-dimensional array, convert to two-dimensional array
    index.add(embeddings.astype(np.float32))
    documents.extend(chunks)


def search(query, top_k=5):
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    results = [(distances[0][i], documents[idx]) for i, idx in enumerate(indices[0])]
    return sorted(results, key=lambda x: x[0])


def call_cloud_model(prompt, folder_path=None):
    global_prompt = (
        "You are an expert with knowledge in the field of hydrogenation reaction, and you will answer a series of questions in the field of hydrogenation according to user requirements and knowledge base content, including experimental scheme design.")

    context = [{"role": "system", "content": global_prompt}]

    context.append({"role": "user", "content": prompt})

    url = "https://api.siliconflow.cn/v1/chat/completions"  #The instance invocation is for the large model api of silicon-based flow. You can switch to another large language model api as needed
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": context,
        "stream": False,
        "max_tokens": 4096,
        "stop": ["exit"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": "Enter your api here",  #Enter your api here
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(directory, filename)
            text = extract_text_from_txt(txt_path)
            chunks = split_into_chunks(text)
            vectorize_and_store(chunks)


def main():
    txt_directory = "./extracted_data"  # Replace with txt text in your knowledge base
    process_directory(txt_directory)

    while True:
        user_query = input("Enter your question (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        results = search(user_query)
        context = "\n".join([doc for _, doc in results])

        prompt = f"Answer the user's question based on the following document fragments:\n{context}\n\nQuestion: {user_query}"
        answer = call_cloud_model(prompt)
        print(f"Answer: {answer}")
        print(f"Source: Document fragments")


if __name__ == "__main__":
    main()



