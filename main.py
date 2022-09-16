from flask import Flask, request, render_template
from csv import writer
from time import time
from validators import url
from nltk.tokenize import sent_tokenize
import nltk
from summariser_functions import article_scraper, csv_checker, wiki_scraper, summary_generator, summary_score, log, report
nltk.download('punkt')

# set these as global variables so they work on the flagged page
text_title = ''
summary = ''
score = 0
timer = 0
error = ''
reduction = 0

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def summarise():
    global text_title, summary, score, timer, error, reduction
    # just in case the log file isnt already created, this will do it
    try:
        file = open('log.csv', 'r')
    except FileNotFoundError:
        file = open('log.csv', 'w')
        csv_writer = writer(file)
        csv_writer.writerow(['text title', 'summary', 'score', 'generation time', 'datetime'])
    file.close()
    start = time()
    # request text input from index.html
    text_inputted = request.form['text']
    # if the text is a url, scrape the article, otherwise scrape the wiki article for the text
    text, text_title = article_scraper(text_inputted) if url(text_inputted) else wiki_scraper(text_inputted)

    # if no article or wiki page could be found for the input, go to error page
    if text_title == 'Error':
        return render_template('error.html', text=text_inputted, error=text)

    # checks if summary already exists in log file
    print(f'summarising {text_title}')
    summary = None
    summary, score = csv_checker(text_title)
    if not summary:
        summary = summary_generator(text)
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
    
    # calculates reduction in size from the original text to the summary
    reduction = round(((len(text) - len(summary)) / len(text)) * 100, 2)

    error = ''

    # if the rouge score is under 0.4, it will return the summary, along with an error message stating the low rouge score
    # (this number is arbitrarily picked, might be a good choice, might not be, who knows???)
    if score < 0.4:
        error = f'Error - Summary generated may not be accurate (Rouge-2 F1 score of {score})'
    return render_template('summary.html', summary=summary, text_title=text_title, time=timer, score=round(score, 2), reduction=reduction, error=error)

# page doesnt work - the summary and stats dont appear
@app.route('/flag', methods=['POST'])
def flag():
    global text_title, summary, score, timer, error, reduction
    # just incase the flagged file hasnt been created, this will do it
    try:
        file = open('flagged.csv', 'r')
    except FileNotFoundError:
        file = open('flagged.csv', 'w')
        csv_writer = writer(file)
        csv_writer.writerow(['text title', 'summary', 'score', 'flag'])
    file.close()
    flag = request.form['flag']
    report(text_title, summary, score, flag)
    print(error)
    return render_template('flag.html', summary=summary, text_title=text_title, time=timer, score=round(score, 2), error=error, reduction=reduction)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)