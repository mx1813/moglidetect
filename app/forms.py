from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    detectmethod = SelectField('Erkennungsmethode', choices=[('compression', 'Kompressionsverfahren'), ('ft', 'Fine-Tuned GBERT'), ('ensemble', 'Alle')], validators=[DataRequired()])
    inputText = TextAreaField('EingabeText', validators=[DataRequired(), Length(min=1, max=512)])
    submit = SubmitField('Check Text!')
