import os
from flask import Flask, render_template, send_from_directory, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretKey'

class InputForm(FlaskForm):
    detectmethod = SelectField('Erkennungsmethode', choices=[('compression', 'Kompressionsverfahren'), ('ft', 'Fine-Tuned GBERT'), ('ensemble', 'Alle')], validators=[DataRequired()])
    inputText = TextAreaField('EingabeText', validators=[DataRequired(), Length(min=1, max=2048)])
    submit = SubmitField('Check Text!')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if form.validate_on_submit():
        detctMethod = form.detectmethod.data
        input = form.inputText.data
        return f'Gewählte Erkennungsmethode: {detctMethod} <br> Zu überprüfender Text: {input}'
    return render_template('index.html', form=form)

@app.route('/api/login', methods=['POST'])
def login():
    form = InputForm()
    if form.validate_on_submit():
        detctMethod = form.detectmethod.data
        input = form.inputText.data
        return f'Gewählte Erkennungsmethode: {detctMethod} <br> Zu überprüfender Text: {input}'
    return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()