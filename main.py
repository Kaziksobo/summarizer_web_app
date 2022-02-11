from flask import Flask, request, render_template
from time import time
from validators import url
from nltk.tokenize import sent_tokenize
from summariser_functions import article_scraper, csv_checker, wiki_scraper, summary_generator, summary_score, log, report

text_title = ''
summary = ''
score = 0
timer = 0
error = ''

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def summarise():
    global text_title, summary, score, timer, error
    f = open('log.csv', 'r+')
    f.close()
    start = time()
    # request text input from index.html
    text = request.form['text']
    # if the text is a url, scrape the article, otherwise scrape the wiki article for the text
    text, text_title = article_scraper(text) if url(text) else wiki_scraper(text)

    print(f'summarising {text_title}')
    summary = None
    summary, score = csv_checker(text_title)
    if not summary:
        summary = summary_generator(text, text_title)
        # fixing summary format
        summary = sent_tokenize(summary)
        if summary[-1][-1] != '.':
            summary.pop()
        summary = ' '.join(summary)
        summary = summary.replace('; ', '')

        score = round(summary_score(summary, text), 2)

    end = time()
    timer = round(end - start, 2)
    print(f'summary generated in {timer}s')
    log(text_title, summary, score, timer)
    reduction = round(((len(text) - len(summary)) / len(text)) * 100, 2)

    error = ''

    # if the rouge score is under 0.4, it will return the summary, along with an error message stating the low rouge score
    # (this number is arbitrarily picked, might be a good choice, might not be, who knows???)
    if score < 0.4:
        error = f'Error - Summary generated may not be accurate (Rouge-2 F1 score of {score})'
    return render_template('summary.html', summary=summary, text_title=text_title, time=timer, score=round(score, 2), reduction=reduction, error=error)

@app.route('/flag', methods=['POST'])
def flag():
    global text_title, summary, score, timer, error
    f = open('flagged.csv', 'r+')
    f.close()
    flag = request.form['flag']
    report(text_title, summary, score, flag)
    print(error)
    return render_template('flag.html', summary=summary, text_title=text_title, time=timer, score=round(score, 2), error=error)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)