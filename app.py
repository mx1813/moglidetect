import os
import json
import time
import random
from flask import Flask, render_template, request, jsonify, url_for
from flask_wtf import FlaskForm
from flask_executor import Executor, futures
from wtforms import SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, InputRequired
from comprendetect import comprendetect


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or 'secretKey'

executor = Executor(app)

def compressionDetection(inputText):
    print("start compression detection")
    print(inputText)
    jsonString = comprendetect.EnsembledZippy().run_on_text_chunked(inputText)
    jsonObject = json.loads(jsonString)
    certValue = round(jsonObject["certainty"], 4)
    certainty = round(certValue * 100, 2)
    certainty = round(certValue * 100, 2) + 50
    if certainty > 100:
            certainty = 99.87
    responseList = {'result':jsonObject["label"], 'certainty':certainty}
    responseJson = json.dumps(responseList)
    print(responseJson)
    return responseJson

@app.route('/start_task')
def long_task(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': 42}
      
class InputForm(FlaskForm):
    detectmethod = SelectField('Methode', choices=[('1', 'Kompression'), ('ft', 'Fine-Tuned GBERT'), ('ensemble', 'Alle')], validators=[DataRequired()])
    inputText = TextAreaField('EingabeText', validators=[InputRequired(message="Du musst einen Text eingeben!"), Length(min=1, max=2048)])
    submit = SubmitField('Check Text!')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if request.method == "POST" and form.validate_on_submit():
        detectMethod = form.detectmethod.data
        input = form.inputText.data
        certainty = 0
        result = 'Loading'
        print(detectMethod)
        if detectMethod == '1':
            print('Detection with Compression Method')
            #task = long_task.apply_async()
            executor.submit_stored('compression', compressionDetection, input)
            print('Started executor')
        else:
            result = 'not yet implemented'
        return render_template('index.html', form=form, result=result, certainty=certainty)
    return render_template('index.html', form=form)

# @app.route('/update-status/<int:state>', methods=['PUT'])
# def update_status(state: int):
#     global status
#     status += 30
#     return '', 200

@app.route('/get-result')
def get_result():
    if not executor.futures.done('compression'):
        state = executor.futures._state('compression')
        return jsonify({'status': state})
    future = executor.futures.pop('compression')
    return jsonify({'status': 'done', 'result': future.result()})



if __name__ == "__main__":
    app.run()