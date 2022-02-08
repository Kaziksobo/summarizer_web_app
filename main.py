from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def hello():
    text = request.form['text']
    text = text.upper()
    return render_template('summary.html', summary=text, text_title=text)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)