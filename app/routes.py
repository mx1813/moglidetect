from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Hugo Habicht'}
    posts = [
        {
            'author': {'username': 'MerdGüller'},
            'body': 'Der Review-Bomber der Nation meldet sich zurück!'
        },
        {
            'author': {'username': 'der71er'},
            'body': 'Knorke Sache mein Freund!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remeber_me={}'.format(
            form.username.data, form.remember_me.data, form.detectmethod.data, form.inputText.data))
        return redirect(url_for('index'))
    return render_template('login.html', title='Log In', form=form)