#!/usr/bin/env python
# encoding: utf-8
from percentile_calculator import stats
from aim_analyzer import analyze_aim
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename

import os

UPLOAD_FOLDER = 'replays'
ALLOWED_EXTENSIONS = {'osr'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ppp')
def ppp():
    return render_template('ppp.html')

@app.route('/ppp', methods=['POST'])
def ppp_post():
    username = request.form['username']
    img1, img2 = stats(username)
    return render_template('ppp-result.html', img1=img1, img2=img2)

@app.route('/aim_analysis')
def aim_analysis():
    return render_template('aim-analysis.html')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/aim_analysis', methods=['POST'])
def aim_analysis_post():
    replay = request.files['replay']
    filename = secure_filename(replay.filename)
    replay.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img1, img2, img3 = analyze_aim(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('aim-analysis-result.html', img1=img1, img2=img2, img3=img3)

if __name__ == '__main__':
    app.run(debug=True)