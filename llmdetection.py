import json
from transformers import pipeline

def llm_pipeline(inputText):
    pipe = pipeline("text-classification", model="./fine-tuning/model/mogli02_sec")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]

def llm_pipeline_dbmz(inputText):
    pipe = pipeline("text-classification", model="./fine-tuning/model/mogli_sec")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]