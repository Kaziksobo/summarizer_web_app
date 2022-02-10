from flask import Flask, request, render_template
from validators import url
from nltk.tokenize import sent_tokenize
from summariser_functions import article_scraper, wiki_scraper, summary_generator, summary_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def hello():
    # request text input from index.html
    text = request.form['text']
    # if the text is a url, scrape the article, otherwise scrape the wiki article for the text
    text, text_title = article_scraper(text) if url(text) else wiki_scraper(text)

    summary = summary_generator(text)

    # fixing summary format
    summary = sent_tokenize(summary)
    if summary[-1][-1] != '.':
        summary.pop()
    summary = ' '.join(summary)
    summary = summary.replace('; ', '')

    score = summary_score(summary, text)

    # if the rouge score is under 0.4, it will return the summary, along with an error message stating the low rouge score
    # (this number is arbitrarily picked, might be a good choice, might not be, who knows???)
    if score < 0.4:
        summary = f'Error - Summary generated may not be accurate (Rouge-2 F1 score of {score})\n\n' + summary


    return render_template('summary.html', summary=summary, text_title=text_title)

    # text = request.form['text']
    # error = None
    # return render_template('summary.html', summary=text.upper(), text_title=text)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)