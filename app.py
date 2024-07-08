import os
import json
import time
from flask import Flask, render_template, jsonify, request, Response
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length
from comprendetect import comprendetect

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY") or 'secretKey'

class InputForm(FlaskForm):
    detectmethod = SelectField('Methode', choices=[('1', 'Kompression'), ('ft', 'Fine-Tuned GBERT'), ('ensemble', 'Alle')], validators=[DataRequired()])
    inputText = TextAreaField('EingabeText', validators=[DataRequired(), Length(min=1, max=2048)])
    submit = SubmitField('Check Text!')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if request.method == "POST":
        if form.validate_on_submit():
            detectMethod = form.detectmethod.data
            input = form.inputText.data
            certainty = 0
            print(detectMethod)
            if detectMethod == '1':
                print('Detection with Compression Method')
                jsonString = comprendetect.EnsembledZippy().run_on_text_chunked(input)
                jsonObject = json.loads(jsonString)
                certainty = str(round(jsonObject["certainty"], 4))
                result = jsonObject["label"]
            else:
                result = 'not yet implemented'
            return render_template('index.html', form=form, result=result, certainty=certainty)
        else:
            return jsonify(data=form.errors)
    return render_template('index.html', form=form)

@app.route('/progress')
def progress():
	def generate():
		x = 0
		
		while x <= 100:
			yield "data:" + str(x) + "\n\n"
			x = x + 10
			time.sleep(0.5)

	return Response(generate(), mimetype= 'text/event-stream')

if __name__ == "__main__":
    app.run()