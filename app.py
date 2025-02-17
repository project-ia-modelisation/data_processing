from fastapi import FastAPI
import json

app = FastAPI()

@app.post("/preprocess")
async def preprocess(data: dict):
    """
    Exemple : prétraitement d'un texte (ajoute '_processed')
    """
    processed_data = {"text": data["text"] + "_processed"}
    return processed_data
