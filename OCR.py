import os
from pathlib import Path
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('Mistral_New')
client = Mistral(api_key=api_key)

file_path = "Hexaware.pdf"

uploaded_file = client.files.upload(
    file={
        "file_name": file_path,
        "content": open(file_path,"rb"),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url),
                                  model="mistral-ocr-latest",
                                  include_image_base64=True)

#pdf_response

response_dict = json.loads(pdf_response.json())
json_string = json.dumps(response_dict, indent=4)

print(json_string)

from mistralai.models import OCRResponse
from IPython.display import Markdown, display

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
  markdowns: list[str] = []
  for page in pdf_response.pages:
    image_data = {}
    for img in page.images:
      image_data[img.id] = img.image_base64
    markdowns.append(replace_images_in_markdown(page.markdown, image_data))

  return "\n\n".join(markdowns)

#display(Markdown(get_combined_markdown(pdf_response)))

#display(Markdown(get_combined_markdown(pdf_response)))

import json
import re
import requests

# --- Configuration ---

def split_text(text, chunk_size=512, overlap=128):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks

def classify_with_groq(text_chunk):
    client = OpenAI(base_url="http://localhost:1247/v1",
                api_key="lm-studio")
    
    system_prompt = (
        "Classify the following text into one of the following categories: "
        "NA (Useless information like disclaimer), Financial Data, Company Information , Future Insight. "
        "Also provide a short summary of the text. "
        "Respond in JSON with keys: text, classification, summary."
    )
    completion = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_chunk}
    ],
    temperature=0.7,
    )

    response = completion.choices[0].message
    print(f"Response: {response}")
    return response

def process_ocr_pages_with_groq(json_string):
    ocr_json = json.loads(json_string)
    results = []
    for page in ocr_json['pages']:
        # Remove image markdown from text, if present
        text = re.sub(r'!\[.*?\]\(.*?\)', '', page['markdown']).strip()
        text_chunks = split_text(text)
        classified_chunks = []
        for chunk in text_chunks:
            try:
                llm_response = classify_with_groq(chunk)
                # LLM expected to return a JSON string
                classified_chunks.append(json.loads(llm_response))
            except Exception as e:
                classified_chunks.append({
                    "text": chunk,
                    "classification": "ERROR",
                    "summary": f"Error: {str(e)}"
                })
        results.append({
            "index": page['index'],
            "classified_chunks": classified_chunks
        })
    return results

# --- Example usage ---
result = process_ocr_pages_with_groq(json_string)
print(json.dumps(result, indent=2))

