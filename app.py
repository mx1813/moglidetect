import os
import json
from typing import TypeAlias
import random
from flask import Flask, render_template, request, jsonify, url_for
from flask_wtf import FlaskForm
from flask_executor import Executor, futures
from wtforms import SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, InputRequired
from comprendetect import comprendetect
from llmdetection import llm_pipeline
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

def zeroShotDetection(inputText):
    print('start zero shot detection')
    detector = AIOrHumanScorer()
    result, n_tokens = detector.score(inputText)
    print(result)
    if result < 0.4:
        result += 0.12
        if result >= 1:
            result = 0.9
        responseList = {'label': 'Mensch', 'score': int((1-result)*100), 'tokens': n_tokens, 'method':3}
    else:
        result += 0.12
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
    if compressionResult["label"] == 'KI':
        weightedVotes.append(-1 * 0.7)
        ssum -= score
    else:
        weightedVotes.append(0.7)
        ssum += score

    llmResult = llm_pipeline(inputText)
    score = llmResult["score"]
    scores.append(score)
    if llmResult["label"] == 'Fake':
        weightedVotes.append(-1*0.3)
        ssum -= score 
    else:
        weightedVotes.append(0.3)
        ssum += score
    
    print(scores)
    print(ssum)
    avg = sum(weightedVotes)/len(weightedVotes)
    print(avg)
    print(sum(scores))
    certainty = abs(ssum)/sum(scores)
    if abs(avg) != 0.5:
        certainty = ssum
    print(certainty)
    if avg < 0:
        responseList = {'label': 'KI', 'score':round(abs(certainty)*100, 2)}
    else:
        responseList = {'label': 'Mensch', 'score':round(abs(certainty)*100, 2)}
    return json.dumps(responseList)
      
class InputForm(FlaskForm):
    detectmethod = SelectField('Methode', choices=[('1', 'Kompression'), ('2', 'Fine-Tuned GBERT'), ('3', 'Zero-shot Detection'), ('4', 'Ensemble')], validators=[DataRequired()])
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
            print('Detection with fine-tuned LLM')
            executor.submit_stored('detection', llmDetection, input)
            print('Started executor')
        elif detectMethod == '3':
            print('Zero-shot detection with google-bert/bert-base-german-cased')
            executor.submit_stored('detection', zeroShotDetection, input)
            print('Started executor')
        else:
            print('Start ensemble detection')
            executor.submit_stored('detection', ensembleDetection, input)
            print('Started executor')
        return render_template('index.html', form=form, method=detectMethod)
    return render_template('index.html', form=form)

@app.route('/get-result')
def get_result():
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    if not executor.futures.done('detection'):
        state = executor.futures._state('detection')
        message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        return jsonify({'status': state, 'message': message})
    future = executor.futures.pop('detection')
    return jsonify({'status': 'done', 'result': future.result()})



if __name__ == "__main__":
    app.run()