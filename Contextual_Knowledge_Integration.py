import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Load locally pretrained embedding model
model = SentenceTransformer('./models/all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Store document chunks and their corresponding IDs
documents = []


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()  # Remove leading and trailing whitespace characters
    print(f"Extracted text length: {len(text)}")  # Debugging information
    return text


def split_into_chunks(text, chunk_size=1000):
    words = text.split()
    print(f"Number of words: {len(words)}")  # Debugging information
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # Ensure chunk is not empty
            chunks.append(chunk)
    print(f"Number of chunks: {len(chunks)}")  # Debugging information
    return chunks


def vectorize_and_store(chunks):
    if not chunks:
        print("No chunks to vectorize.")  # Debugging information
        return

    embeddings = model.encode(chunks)
    print(f"Embeddings shape: {embeddings.shape}")  # Debugging information
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)  # If it's a one-dimensional array, convert it to a two-dimensional array
    index.add(embeddings.astype(np.float32))
    documents.extend(chunks)


def search(query, top_k=5):
    query_embedding = model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    results = [(distances[0][i], documents[idx]) for i, idx in enumerate(indices[0])]
    return sorted(results, key=lambda x: x[0])


def call_cloud_model(prompt, folder_path=None):
    global_prompt = ("You are an expert chemist! Read the provided literature file about reactions carefully. "
                     "Compile a detailed table using the provided paragraphs, and extract and present the following details:\n"
                     "a. reactant\nb. temperature\nc. pressure\nd. reactant concentration\ne. catalyst\nf. reagent\ng. product\nh. complete reaction")

    context = [{"role": "system", "content": global_prompt}]

    if folder_path:
        initial_prompt = (
                "Extract chemical reactions from all TXT files in the following folder and compile them into a table. "
                "Folder path:" + folder_path)
        context.append({"role": "user", "content": initial_prompt})
        print(f"Initial prompt sent.\n")

    context.append({"role": "user", "content": prompt})

    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": context,
        "stream": False,
        "max_tokens": 512,
        "stop": ["exit"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
        # Removed "tools" parameter
    }
    headers = {
        "Authorization": "input your api",  # Input your own API key here
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
            print(f"Processing TXT: {filename}")
            text = extract_text_from_txt(txt_path)
            chunks = split_into_chunks(text)
            vectorize_and_store(chunks)


def main():
    txt_directory = "api_result"  # Replace with your TXT folder path
    process_directory(txt_directory)

    # Send initial request to extract chemical reaction information
    call_cloud_model("", folder_path=txt_directory)

    while True:
        user_query = input("Please enter your question (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        results = search(user_query)
        context = "\n".join([doc for _, doc in results])

        prompt = f"Answer the user's question based on the following document snippets:\n{context}\n\nQuestion: {user_query}"
        answer = call_cloud_model(prompt)
        print(f"Answer: {answer}")
        print(f"Source: Document snippets")


if __name__ == "__main__":
    main()