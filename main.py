from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
import torch
import uvicorn
import re
import logging
from time import time

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_analysis_app")

# FastAPI uygulaması oluştur
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Middleware ile her istek için loglama
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    start_time = time()
    
    response = await call_next(request)
    
    process_time = time() - start_time
    logger.info(f"Completed in {process_time:.4f} seconds")
    
    return response

# Modellerin ve tokenizerların yüklenmesi
sentiment_model_name = "mesutaktas/TurkishNERModel"
ner_model_name = "savasy/bert-base-turkish-ner-cased"

sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name)

ner_tokenizer = BertTokenizer.from_pretrained(ner_model_name)
ner_model = BertForTokenClassification.from_pretrained(ner_model_name)

def sentiment_analysis(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()
    sentiment_map = {0: "olumsuz", 1: "nötr", 2: "olumlu"}
    return sentiment_map[sentiment]

def clean_entity(entity):
    # Anlamsız entity'leri temizle (tek karakterli, sayılar, vs.)
    if len(entity) <= 2 or entity.lower() in ["##", "lar", "lar", "ler", "zken", "dığım", "eyim"]:
        return None
    entity = re.sub(r'[^\w\s]', '', entity)  # Noktalama işaretlerini kaldır
    return entity

def named_entity_recognition(text):
    inputs = ner_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = ner_model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    label_map = ner_model.config.id2label
    entities = []
    current_entity = ""

    for token, prediction in zip(tokens, predictions[0]):
        label = label_map[prediction.item()]
        if label != "O":  # "O" olmayanları dikkate al
            if token.startswith("##"):
                current_entity += token[2:]
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = ""
    
    if current_entity:
        entities.append(current_entity)

    # Temizle ve filtrele
    clean_entities = []
    for entity in entities:
        clean_entity_text = clean_entity(entity)
        if clean_entity_text:
            clean_entities.append(clean_entity_text)

    return clean_entities

def analyze_text(text):
    entities = named_entity_recognition(text)
    results = []
    for entity in entities:
        sentiment = sentiment_analysis(entity)
        results.append({"entity": entity, "sentiment": sentiment})
    
    output = {
        "entity_list": entities,
        "results": results
    }
    return output

# POST request ile gelen veriyi işleyen endpoint
class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(input: TextInput):
    result = analyze_text(input.text)
    logger.info(f"Prediction result: {result}")
    return result

# HTML sayfasını sunan GET endpoint
@app.get("/", response_class=HTMLResponse)
def get_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Tool</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-image: url('/static/background.webp'); /* Resmin yolu buraya */
                background-size: cover; /* Resmin tüm arka planı kaplamasını sağlar */
                background-position: center; /* Resmin ortalanmasını sağlar */
                background-repeat: no-repeat; /* Resmin tekrar etmemesini sağlar */
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background-color: rgba(255, 255, 255, 0.8); /* Arka plan renginin biraz şeffaf olması */
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                width: 90%;
                max-width: 700px;
                padding: 20px;
                box-sizing: border-box;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 20px;
                font-size: 24px;
            }
            h2 {
                color: #555;
                margin-top: 0;
                font-size: 20px;
            }
            label {
                font-size: 16px;
                color: #555;
            }
            textarea {
                width: calc(100% - 20px);
                padding: 10px;
                margin-top: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 16px;
                box-sizing: border-box;
                resize: vertical; /* Kullanıcı sadece dikey yönde boyutlandırabilir */
            }
            input[type="button"] {
                width: 100%;
                padding: 10px;
                border: none;
                border-radius: 4px;
                background-color: #007bff;
                color: white;
                font-size: 18px;
                cursor: pointer;
                margin-top: 10px;
            }
            input[type="button"]:hover {
                background-color: #0056b3;
            }
            .output {
                background-color: #f8f9fa;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 20px;
                max-height: 300px;
                overflow-y: auto; /* İçeriğin taşmasını kaydırma çubuğu ile gösterir */
            }
            .output pre {
                margin: 0;
                white-space: pre-wrap; /* Satır sonlarını korur */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sentiment Analysis Tool</h1>
            <h2>Analyze Text Sentiment</h2>
            <form id="predict-form">
                <label for="text">Input Text:</label>
                <textarea id="text" name="text" rows="8" placeholder="Type your text here..." required></textarea>
                <input type="button" value="Analyze" onclick="submitForm()">
            </form>
            <div class="output">
                <h2>Analysis Result:</h2>
                <pre id="result"></pre>
            </div>
        </div>
        <script>
            function submitForm() {
                const text = document.getElementById('text').value;
                fetch('/predict/', {
                    method: 'POST',
                    body: JSON.stringify({ text }),
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# FastAPI uygulamasını başlatmak için
if __name__ == "__main__":
    # Uvicorn sunucusunu başlat
    uvicorn.run(app, host="0.0.0.0", port=1352)
