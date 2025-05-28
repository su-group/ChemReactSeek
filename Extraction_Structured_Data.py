import fitz  # PyMuPDF
import os
import requests
import time
import shutil


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def parse_experiment_data(text, api_url, headers):
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "user", "content": f"Extract structured experimental data from the following text:\n{text}"},
            {"role": "assistant", "content": """
Please provide the extracted experimental data in the following format:

Chemical Reaction:
...

Experimental Conditions:
Reagents: ...
Solvent: ...
Catalyst: ...
Reaction Temperature: ...
Reaction Time: ...
Reaction Pressure: ...
pH Value: ...

Specific Values:
... (include all specific values mentioned)

Yield Prediction:
Expected Yield: ...

Procedure:
1. ...
2. ...
3. ...
...

Notes:
...
"""}
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": ["exit"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }

    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    elif response.status_code == 429:
        raise Exception(f"API request failed with status code {response.status_code}: Rate limit exceeded")
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


def save_to_txt(data, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
        file.write(data)


def move_processed_pdf(source_path, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    shutil.move(source_path, os.path.join(destination_folder, os.path.basename(source_path)))


def process_pdfs_in_folder(folder_path, output_dir, processed_dir, api_url, headers, initial_delay=10, max_retries=5):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        retries = 0
        while retries < max_retries:
            try:
                parsed_data = parse_experiment_data(text, api_url, headers)
                txt_filename = os.path.splitext(pdf_file)[0] + '.txt'
                save_to_txt(parsed_data, output_dir, txt_filename)
                print(f"Saved extracted data to {os.path.join(output_dir, txt_filename)}")
                move_processed_pdf(pdf_path, processed_dir)
                print(f"Moved {pdf_file} to {processed_dir}")
                break
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
                if "Rate limit exceeded" in str(e):
                    wait_time = initial_delay * (2 ** retries)
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    break
        if retries == max_retries:
            print(f"Max retries reached for {pdf_file}. Skipping this file.")
        time.sleep(initial_delay)  # Add delay to avoid rate limiting


# Example usage
folder_path = 'papers'  # Path to the folder containing PDF files
output_dir = 'api_result'  # Target directory where extracted data will be saved
processed_dir = 'api_processed'  # Directory where processed PDF files will be moved
api_url = "input your api url"  #Please enter the API endpoint URL here, referring specifically to the documentation provided by the API service provider.
headers = {
    "Authorization": "input your api",  # Input your own API key here
    "Content-Type": "application/json"
}

process_pdfs_in_folder(folder_path, output_dir, processed_dir, api_url, headers)