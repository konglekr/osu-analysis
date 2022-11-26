#!/usr/bin/env python
# encoding: utf-8
from percentile_calculator import stats
from flask import request, render_template
from flask import Flask

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    username = request.form['username']
    img1, img2 = stats(username)
    return render_template('layout.html', img1=img1, img2=img2)

if __name__ == '__main__':
    app.run(debug=True)