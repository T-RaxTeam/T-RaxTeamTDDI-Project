import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.nn.functional import softmax
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict
from pydantic import BaseModel
from datasets import Dataset
import uvicorn
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import re
import traceback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
app = FastAPI()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
class Item(BaseModel):
    text: str
@app.get("/")
async def read_root():
    return {"message": "Yorum Duygu Analizi API'sine hoş geldiniz!"}
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global tokenizer, model
    try:
        df = pd.read_excel(file.file)
        
        if 'Yorum' not in df.columns:
            raise HTTPException(status_code=400, detail="Dosyada 'Yorum' sütunu bulunamadı.")
        
        if df['Yorum'].isnull().any():
            raise HTTPException(status_code=400, detail="Veri setinde boş cümleler var. Lütfen kontrol edin.")
        
        df['label'] = 0.0  # Burada uygun etiketleri ayarlayın
        train_df = df.sample(frac=0.8, random_state=42)
        eval_df = df.drop(train_df.index)
        train_dataset = Dataset.from_pandas(train_df[['Yorum', 'label']])
        eval_dataset = Dataset.from_pandas(eval_df[['Yorum', 'label']])
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples):
            return tokenizer(examples['Yorum'], padding="max_length", truncation=True, max_length=128)
        tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)
        def convert_labels_to_float(example):
            example['label'] = float(example['label'])
            return example
        tokenized_train_datasets = tokenized_train_datasets.map(convert_labels_to_float)
        tokenized_eval_datasets = tokenized_eval_datasets.map(convert_labels_to_float)
        
        num_labels = len(df['label'].unique())
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_eval_datasets,
            compute_metrics=compute_metrics  # Bu satırı ekleyin
        )
        
        trainer.train()
        model.save_pretrained("sentiment_model")
        tokenizer.save_pretrained("sentiment_tokenizer")
        
        return {"message": "Model başarıyla eğitildi"}
    except Exception as e:
        error_details = traceback.format_exc()
        print("Eğitim sırasında hata:", error_details)
        return JSONResponse(content={"detail": str(e)}, status_code=500)

@app.post("/analyze")
async def analyze_sentiment(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)  # CSV dosyasını oku

        # Eksik sıra numaralarını tamamla
        if 'Sıra No' in df.columns:
            df['Sıra No'] = range(1, len(df) + 1)
        
        # Boş cümleleri kontrol et
        if df['Yorum'].isnull().any():
            raise HTTPException(status_code=400, detail="Dosyada boş cümleler var. Lütfen kontrol edin.")
        
        yorumlar = df['Yorum'].tolist()
        results = []
        
        for yorum in yorumlar:
            words = re.findall(r'\b\w+\b', yorum)  # Kelimeleri ayır
            for word in words:
                inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True, max_length=128)
                outputs = model(**inputs)
                probs = softmax(outputs.logits, dim=1)
                sentiment = torch.argmax(probs, dim=1).item()
                sentiment_score = probs[0][sentiment].item()

                labels = ["Çok Negatif", "Negatif", "Nötr", "Pozitif", "Çok Pozitif"]
                etiket = labels[sentiment]
                
                results.append({"kelime": word, "etiket": etiket, "skor": sentiment_score})
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(item: Item):
    text = item.text
    words = re.findall(r'\b\w+\b', text)  # Kelimeleri ayır
    results = []

    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        sentiment = torch.argmax(probs, dim=1).item()
        sentiment_score = probs[0][sentiment].item()

        labels = ["Çok Negatif", "Negatif", "Nötr", "Pozitif", "Çok Pozitif"]
        etiket = labels[sentiment]
        
        results.append({
            "entity": word,
            "sentiment": etiket,
            "score": sentiment_score
        })

    return {
        "entity_list": words,
        "results": results
    }

@app.get("/predict-form")
async def predict_form():
    return HTMLResponse("""
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
    """)

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8080)

def open_url():
    url = "http://127.0.0.1:8080/predict-form"
    response = requests.get(url)
    if response.status_code == 200:
        import webbrowser
        webbrowser.open(url)
    else:
        messagebox.showerror("Hata", "URL'yi açma başarısız oldu")

def start_tkinter():
    root = tk.Tk()
    root.title("Model Eğitim Sayfası")
    
    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (file_path, f)}
                    response = requests.post("http://127.0.0.1:8080/train", files=files)
                    if response.status_code == 200:
                        messagebox.showinfo("Başarılı", "Model başarıyla eğitildi")
                    else:
                        messagebox.showerror("Hata", f"Hata: {response.text}")
            except Exception as e:
                messagebox.showerror("Hata", f"Hata: {str(e)}")
    
    train_button = tk.Button(root, text="Model Eğit", command=open_file)
    train_button.pack(pady=20)
    
    sentiment_button = tk.Button(root, text="Duygu Analizi Yap", command=open_url)
    sentiment_button.pack(pady=10)
    
    root.mainloop()

# FastAPI'yi farklı bir iş parçacığında çalıştır
api_thread = threading.Thread(target=run_fastapi)
api_thread.start()

# Tkinter arayüzünü başlat
start_tkinter()
