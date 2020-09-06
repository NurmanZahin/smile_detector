from datetime import datetime
from flask import Flask, request, render_template, redirect
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
db = SQLAlchemy(app)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print('not implemented')
