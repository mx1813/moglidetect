from flask import render_template
from app import app

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