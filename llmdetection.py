import json
from transformers import pipeline

def llm_pipeline(inputText):
    pipe = pipeline("text-classification", model="./fine-tuning/model/mogli_gbert")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]

def llm_pipeline_dbmz(inputText):
    pipe = pipeline("text-classification", model="./fine-tuning/model/mogli_dbmz")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]