from flask import Flask, render_template, request
import json
from imagen import save_num

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate/', methods=['POST'])
def send_images():
    number = str(request.form['number'])
    gen = str(request.form['gen'])
    if not number.isdigit():
        return 'Incorrect number', 400
    number = int(number)
    if not gen.isdigit():
        return 'Incorrect number of generations', 400
    gen = int(gen)
    if gen > 3:
        return 'Incorrect number of generations', 400
    files = save_num(number, gen)
    return json.dumps({'images':files})