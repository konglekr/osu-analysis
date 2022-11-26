#!/usr/bin/env python
# encoding: utf-8
from percentile_calculator import stats
from flask import request, render_template
from flask import Flask
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
bootstrap = Bootstrap5(app)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    username = request.form['username']
    stats(username)
    return render_template('layout.html')

app.run(debug=True)