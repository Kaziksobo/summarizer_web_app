from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime
import wikipedia
import wikipediaapi
from rouge import Rouge
from nltk.tokenize import sent_tokenize
wiki_api = wikipediaapi.Wikipedia('en')

def article_scraper(text):
    # scrapes text of the article using newspaper3k
    try:
        article = Article(text)
        article.download()
        article.parse()
    except:
        return 'That article cannot be summarised', 'Error'
    return article.text.replace('\n', ''), article.title

def wiki_scraper(text):
    # This code is so crap
    # I use both a python wrapper for the wikipedia api and an unmaintened wikipedia module
    try:
        text_title = wikipedia.search(text)[0]
        wikisearch = wiki_api.page(text)
    except:
        return 'Error - that topic cannot be summarised', 'Error'
    return wikisearch.text, text_title

    



def summary_generator(text):
    # creates summary using BART transformer from huggingfaces
    checkpoint = 'facebook/bart-large-cnn'
    # create tokenizer using checkpoint model
    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    print(f'created tokenizer - {datetime.now().time()}')
    #tokenizes inputs using created tokenizer, truncating to the model's max length, using pytorch tensors
    input_ids = tokenizer.batch_encode_plus(
        [text], 
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors='pt'
    )
    print(f'tokenized inputs - {datetime.now().time()}')
    # creates model using BART for conditional generation, pretrained on the checkpoint model
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    print(f'created model - {datetime.now().time()}')
    # selects summary tokens, using beam search with 5 beams, stopping when all beam hypotheses reached the EOS token
    summary_ids = model.generate(
        input_ids['input_ids'],
        num_beams=5,
        early_stopping=True
    )
    print(f'generated summary ids - {datetime.now().time()}')
    # decodes the summary tokens, not counting any special tokens the model may have generated
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
    )
    print(f'Generated summary - {datetime.now().time()}')
    return summary

def summary_score(summary, text):
    reference = ' '.join(sent_tokenize(text)[:3])
    scores = Rouge().get_scores(summary, reference)[0]
    return scores['rouge-2']['f']