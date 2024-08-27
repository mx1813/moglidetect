import pandas as pd
import re, json
import csv
from datasets import Dataset
from llmdetection import llm_pipeline
from zeroShotDetection import AIOrHumanScorer
from comprendetect import comprendetect
from app import compressionDetection


data = pd.read_csv('evaluationDS.csv', on_bad_lines='skip', sep=";", encoding="utf-8")
print(len(data))
vds = Dataset.from_pandas(data)

humans = data.loc[data['label'] == 0]
ais = data.loc[data['label'] == 1]

print(len(humans))
print(len(ais))

evaDs = vds.to_iterable_dataset()
print(evaDs)
tp = []
tn = []
fp = []
fn = []
def clean_text(s : str) -> str:
    '''
    Removes formatting and other non-content data that may skew compression ratios (e.g., duplicate spaces)
    '''
    # Remove extra spaces and duplicate newlines.
    s = re.sub(' +', ' ', s)
    s = re.sub('\t', '', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub('\n ', '\n', s)
    s = re.sub(' \n', '\n', s)

    # Remove non-alphanumeric chars
    s = re.sub(r'[^0-9A-Za-z,\.\(\) \n]', '', s)#.lower()

    return s

with open("evalC.csv", 'w', newline='\n', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quotechar='"', delimiter=';')
    field = ["true_label", "assigned_label", "result"]
    writer.writerow(field)
    # Write each paragraph as a new row in the CSV
    i=1
    # write with default settings
    ass_label = 0
    result = 'TP'
    print(result)
    #detector = AIOrHumanScorer()
    compDetect = comprendetect.EnsembledZippy()
    for text in evaDs:
        print(i)
        inputText = text['text']
        #jsonRes = llm_pipeline(inputText[:2048])
        #result, n_tokens = detector.score(inputText[:2048])
        jsonString = compDetect.run_on_text_chunked(inputText[:2048])
        jsonObject = json.loads(jsonString)
        print(jsonObject)
        if jsonObject['label'] == 'KI':
            ass_label = 1
        else: 
            ass_label = 0
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
        writer.writerow([text['label'], ass_label, result])
        i+=1
        
print("Ergebnisse LLM detection")
print(len(tp))
print(len(fp))
print(len(tn))
print(len(fn))