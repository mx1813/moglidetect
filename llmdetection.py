import json
from transformers import pipeline

def llm_pipeline(inputText):
    pipe = pipeline("text-classification", model="idajikuu/AI-HUMAN-detector")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]