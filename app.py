#----------------------------------------------
# @AUTHOR: cwc\bbobo
# INPUTS MUST BE ONE (or more) OF THE FOLLOWING: 
# 1) path/ARTICLE.txt
# 2) path/CLASSIFIERS.csv path/CLASSIFIERS_TXT.txt
# 3) http URL
#
# OUTPUTS ARE SAVED TO THE SAME PATH AS THE INPUTS (or base folder if URL)
#----------------------------------------------

from transformers import pipeline
from transformers import AutoTokenizer
import os
import sys
import json
import csv
from bs4 import BeautifulSoup
import requests
import base64

#-- Article Summary HF Model using *Transformers Pipeline*
def article_summary(ARTICLE):
    #-- init model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    if len(ARTICLE) > 1024: #-- MAX TOKEN LIMIT FOR MODEL
        ARTICLE = ARTICLE[:1024]
    #-- call model
    ARTICLE = summarizer(ARTICLE, do_sample=False)

    return ARTICLE[0]['summary_text']

#-- Classifier HF Model using *Transformers Pipeline*
def article_classifier(ARTICLE, LABELS):
    #-- init model
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
    classifier = pipeline("zero-shot-classification", model=model_name,tokenizer=tokenizer)

    #-- call model
    output = classifier(ARTICLE, LABELS, multi_label=True)
    return output['scores']


def main():
    ##CONST NAMES FOR INPUTS##
    ARTICLE_FILENAME = "ARTICLE.TXT"
    CLASSIFIER_FILENAME = "CLASSIFIERS.CSV"
    CLASSIFIER_TXT_FILENAME = "CLASSIFIERS_TXT.TXT"
    HTTP_NAME = "HTTP"

    for input in sys.argv:
        #------------------------TEXT ARTICLE LOGIC------------------------
        if ARTICLE_FILENAME in input.upper():
            # get article text
            file_path_article = input.replace('\\','/')
            with open(file_path_article, encoding="utf8", errors="ignore") as f:
                ARTICLE = f.read()

            # run model
            ARTICLE = article_summary(ARTICLE)

            # save summary text
            file_name_article = os.path.basename(file_path_article).split('/')[-1]
            file_path_article = file_path_article.replace(file_name_article,'')
            with open(file_path_article+'OUTPUT_article-summary.txt', 'w') as f:
                f.write(ARTICLE)
            
        #------------------------CLASSIFIER LOGIC------------------------
        if CLASSIFIER_FILENAME in input.upper():
            # get classifiers
            file_path_classifiers = input.replace('\\','/')
            with open(file_path_classifiers, newline='') as f:
                CLASSIFIERS = list(csv.reader(f))[0]
            for input in sys.argv:
                if CLASSIFIER_TXT_FILENAME in input.upper():

                    # get classifier text
                    file_path_article = input.replace('\\','/')
                    with open(file_path_article, encoding="utf8", errors="ignore") as f:
                        ARTICLE = f.read()

                    # run model
                    CLASSIFIERS = article_classifier(ARTICLE, CLASSIFIERS)

                    # save classifications
                    file_name_article = os.path.basename(file_path_article).split('/')[-1]
                    file_path_article = file_path_article.replace(file_name_article,'')
                    with open(file_path_article+'OUTPUT_classifiers.json', 'w') as f:
                        f.write(json.dumps(CLASSIFIERS))
                        
        #------------------------HTTP ARTICLE LOGIC-----------------------
        if HTTP_NAME in input.upper():
            # decode google rss url
            if "GOOGLE.COM/RSS" in input.upper():
                google_url = input
                base64_url = google_url.replace("https://news.google.com/rss/articles/","").split("?")[0]
                 # Replace underscores with slashes and hyphens with plus signs
                base64_url = base64_url.replace("_", "/").replace("-", "+")
                 # Calculate the number of padding '=' characters needed
                padding = (4 - len(base64_url) % 4) % 4
                base64_url += '=' * padding
                # Decode String
                input = base64.b64decode(base64_url).decode('latin-1')
                input = input[input.index('http'):]
                input = input.split('Ò',1)[0]

            # get website text
            response = requests.get(input,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36','cache-control': 'max-age=0'}, cookies={'cookies':''})
            soup = BeautifulSoup(response.text,features="html.parser")

            article = soup.find('article')

            if article is not None:
                ARTICLE = soup.article.get_text(' ', strip=True)
                # save summary text
                with open('OUTPUT_article-summary.txt', 'w') as f:
                    f.write(ARTICLE)
            else: 
                print("No <article> tag found on the page.")
                text_elements = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                if text_elements:
                    ARTICLE = ' '.join([element.get_text(' ', strip=True) for element in text_elements])
                # save summary text
                with open('OUTPUT_article-summary.txt', 'w') as f:
                    f.write(ARTICLE)

if __name__ == '__main__':
    main()