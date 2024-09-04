import pandas as pd
import csv, json
from datasets import Dataset
from llmdetection import llm_pipeline, llm_pipeline_dbmz
from zeroShotDetection import AIOrHumanScorer
from comprendetect import comprendetect
from app import compressionDetection, ensembleDetection


data = pd.read_csv('dsValidate.csv', on_bad_lines='skip', sep=";", encoding="utf-8")
print(len(data))
vds = Dataset.from_pandas(data)
evaDs = vds.to_iterable_dataset()
print(evaDs)
tp = []
tn = []
fp = []
fn = []

with open("evaLLMK.csv", 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, quotechar='"', delimiter=',')
    field = ["true_label", "assigned_label", "score"]
    writer.writerow(field)
    # Write each paragraph as a new row in the CSV
    i=1
    # write with default settings
    ass_label = 0
    result = 'TP'
    print(result)
    #detector = AIOrHumanScorer()
    #compDetect = comprendetect.EnsembledZippy()
    for text in evaDs:
        print(i)
        inputText = text['text']
        jsonObject = llm_pipeline_dbmz(inputText[:2048])
        #result, n_tokens = detector.score(inputText[:2048])
        #jsonString = ensembleDetection(inputText[:2048])
        #jsonObject = json.loads(jsonString)
        #jsonObject = json.loads(comprendetect.EnsembledZippy().run_on_text_chunked(inputText))
        score = jsonObject['score']
        print(jsonObject)
        if jsonObject['label'] == 'AI':
            ass_label = 1
        else: 
            ass_label = 0
            score = 1 - jsonObject['score']
        print(text['label'])
        if text['label'] == 0:
            if ass_label == 0:
                result = 'TN'
                tn.append(i)
            else:
                result = 'FP'
                fp.append(i)
        else:
            if ass_label == 1:
                result = 'TP'
                tp.append(i)
            else:
                result = 'FN'
                fn.append(i)
        print(result)
        writer.writerow([text['label'], ass_label, score])
        i+=1
        
print("Ergebnisse LLM detection")
print(len(tp))
print(len(fp))
print(len(tn))
print(len(fn))