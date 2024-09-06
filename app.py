import os
import json
import numpy as np
import random
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from flask_executor import Executor
from wtforms import SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, InputRequired
from comprendetect import comprendetect
from llmdetection import llm_pipeline, llm_pipeline_dbmz
from zeroShotDetection import AIOrHumanScorer


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or 'secretKey'

executor = Executor(app)

def compressionDetection(inputText):
    print("start compression detection")
    jsonString = comprendetect.EnsembledZippy().run_on_text_chunked(inputText)
    jsonObject = json.loads(jsonString)
    scoreValue = round(jsonObject["score"], 4)
    score = round(scoreValue * 100, 2)
    certainty = round(jsonObject["certainty"], 6)
    responseList = {'label':jsonObject["label"], 'score':score, 'certainty':certainty, 'method':1}
    responseJson = json.dumps(responseList)
    print(responseJson)
    return responseJson

def llmDetection(inputText):
    print('start llm pipeline')
    jsonRes = llm_pipeline(inputText)
    print(jsonRes)
    score = round(jsonRes["score"] * 100, 2)
    responseList = {'label':jsonRes["label"], 'score':score, 'method':2}
    responseJson = json.dumps(responseList)
    return responseJson

def llmDetectionDbmz(inputText):
    print('start llm pipeline')
    jsonRes = llm_pipeline_dbmz(inputText)
    print(jsonRes)
    score = round(jsonRes["score"] * 100, 2)
    responseList = {'label':jsonRes["label"], 'score':score, 'method':5}
    responseJson = json.dumps(responseList)
    return responseJson

def zeroShotDetection(inputText):
    print('start zero shot detection')
    detector = AIOrHumanScorer()
    result, n_tokens = detector.score(inputText)
    print(result)
    if result <= 0.5:
        if result == 0.5:
            result-=0.12
        responseList = {'label': 'Mensch', 'score': int((1-result)*100), 'tokens': n_tokens, 'method':3}
    else:
        if result >= 1:
            result = 0.9
        responseList = {'label': 'KI', 'score': int(result*100), 'tokens': n_tokens, 'method':3}
    print(responseList)
    return json.dumps(responseList)

def ensembleDetection(inputText):
    print('start ensemble detection')
    scores = []
    weightedVotes = []
    ssum: float = 0.0
    compressionResult = json.loads(comprendetect.EnsembledZippy().run_on_text_chunked(inputText))
    print(compressionResult)
    print(compressionResult["score"])
    score = compressionResult["score"]
    scores.append(score)
    # compression detection
    if compressionResult["label"] == 'KI':
        weightedVotes.append(-1 *0.2)
    else:
        weightedVotes.append(0.2)
    # fine tuned gbert detection
    llmResult = llm_pipeline(inputText)
    score = llmResult["score"]
    scores.append(score)
    if llmResult["label"] == 'AI':
        weightedVotes.append(-1*0.3)
    else:
        weightedVotes.append(0.3)
    #fine-tuned dbmz-bert
    llmResult = llm_pipeline_dbmz(inputText)
    score = llmResult["score"]
    scores.append(score)
    if llmResult["label"] == 'AI':
        weightedVotes.append(-1*0.4)
    else:
        weightedVotes.append(0.4)
    zeroShotResult = json.loads(zeroShotDetection(inputText))
    print(zeroShotResult)
    score = zeroShotResult["score"] / 100
    scores.append(score)
    # fine tuned llm detection
    if zeroShotResult["label"] == 'KI':
        weightedVotes.append(-1*0.1)
    else:
        weightedVotes.append(0.1)
    print(scores)
    print(weightedVotes)
    avg = sum(weightedVotes)/len(weightedVotes)
    print(avg)
    # normalize ssum
    certainty = (abs(sum(weightedVotes)) + np.max(scores)) / 2
    print(certainty)
    if abs(certainty)<=0.5:
        certainty=0.5241
    if avg < 0:
        responseList = {'label': 'KI', 'score':round(abs(certainty)*100, 2)}
    else:
        responseList = {'label': 'Mensch', 'score':round(abs(certainty)*100, 2)}
    print(responseList)
    return json.dumps(responseList)
      
class InputForm(FlaskForm):
    detectmethod = SelectField('Methode', choices=[('1', 'Kompression'), ('2', 'Fine-Tuned GBERT'), ('5', 'Fine-Tuned DBMZ-BERT'), ('3', 'Zero-shot Detection'), ('4', 'Ensemble')], validators=[DataRequired()])
    inputText = TextAreaField('EingabeText', validators=[InputRequired(message="Du musst einen Text eingeben!"), Length(min=1, max=2048)])
    submit = SubmitField('Check Text!')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if request.method == "POST" and form.validate_on_submit():
        detectMethod = form.detectmethod.data
        input = form.inputText.data
        print(detectMethod)
        if detectMethod == '1':
            print('Detection with Compression Method')
            #task = long_task.apply_async()
            executor.submit_stored('detection', compressionDetection, input)
            print('Started executor')
        elif detectMethod == '2':
            print('Detection with fine-tuned GBERT')
            executor.submit_stored('detection', llmDetection, input)
            print('Started executor')
        elif detectMethod == '3':
            print('Zero-shot detection with google-bert/bert-base-german-cased')
            executor.submit_stored('detection', zeroShotDetection, input)
            print('Started executor')
        elif detectMethod == '5':
            print('Detection with fine-tuned DBMZ-BERT')
            executor.submit_stored('detection', llmDetectionDbmz, input)
            print('Started executor')
        else:
            print('Start ensemble detection')
            executor.submit_stored('detection', ensembleDetection, input)
            print('Started executor')
        return render_template('index.html', form=form, method=detectMethod)
    return render_template('index.html', form=form)

@app.route('/get-result')
def get_result():
    verb = ['entschlüsseln', 'durchleuchten', 'analysieren', 'durchforsten', 'bewerten', 'entwirren', 'untersuchen']
    adjective = ['komplexe', 'verwirrte', 'geheime', 'intrigante', 'kreative', 'versteckte', 'klare', 'ungewöhnliche', 'interessante', 'spannende']
    noun = ['Inhalte', 'Daten', 'Informationen', 'Sätze', 'Texte', 'Bits und Bytes', 'Code', 'Wörter', 'Algorithmen']
    message = ''
    if not executor.futures.done('detection'):
        state = executor.futures._state('detection')
        message = '{0} {1} {2}...'.format(random.choice(adjective),
                                              random.choice(noun),
                                              random.choice(verb))
        return jsonify({'status': state, 'message': message})
    future = executor.futures.pop('detection')
    return jsonify({'status': 'done', 'result': future.result()})



if __name__ == "__main__":
    app.run()