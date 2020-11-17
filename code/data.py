# Reading in sentiment data to create corpus

from zipfile import ZipFile # for reading amazon review zip
import pandas as pd
import nltk # Natural Language Toolkit
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Method clean_text was taken from data.py in HW4 from CS_475

def clean_text(text):
    """
    Remove stopwords, punctuation, and numbers from text.

    Args:
        text: article text

    Returns:
        Space-delimited and cleaned string
    """
    # tokenize text
    tokens = nltk.word_tokenize(text)
    # remove stopwords
    tokens = [token.lower().strip() for token in tokens if token.lower() not in stopwords]

    # remove tokens without alphabetic characters (i.e. punctuation, numbers)
    tokens = [token for token in tokens if any(t.isalpha() for t in token)]

    return ' '.join(tokens)

def get_sentiment_corpus_and_labels():
    amazon_labels = []
    amazon_sentences = []
    with open('datasets/amazon_sentiments.txt') as amazon_sentiments:
        for line in amazon_sentiments:
            line = line.split()
            text = line[:-1] # get all words except label
            string = " "
            amazon_sentences.append(clean_text(string.join(text))) # get cleaned text data
            amazon_labels.append(line[-1]) # get label
    imdb_labels = []
    imdb_sentences = []
    with open('datasets/imdb_sentiments.txt') as imdb_sentiments:
        for line in imdb_sentiments:
            line = line.split()
            text = line[:-1] # get all words except label
            string = " "
            imdb_sentences.append(clean_text(string.join(text))) # get cleaned text data
            imdb_labels.append(line[-1]) # get label
    yelp_labels = []
    yelp_sentences = []
    with open('datasets/yelp_sentiments.txt') as yelp_sentiments:
        for line in yelp_sentiments:
            line = line.split()
            text = line[:-1] # get all words except label
            string = " "
            yelp_sentences.append(clean_text(string.join(text))) # get cleaned text data
            yelp_labels.append(line[-1]) # get label
    amazon_sentences.extend(imdb_sentences)
    amazon_sentences.extend(yelp_sentences)
    sentiment_corpus = amazon_sentences
    amazon_labels.extend(imdb_labels)
    amazon_labels.extend(yelp_labels)
    sentiment_labels = amazon_labels

    return sentiment_corpus, sentiment_labels

def get_amazon_reviews_corpus_and_labels():
    zip_file = ZipFile('datasets/amazon_reviews.zip')
    raw_dataframe = pd.read_csv(zip_file.open('amazon_reviews.csv'))
    col_names = list(raw_dataframe.columns) # get categories in review dataset
    raw_rating_text = raw_dataframe['reviews.text'] # uncleaned text
    rating_text = []
    for review in raw_rating_text: # clean rating text
        rating_text.append(clean_text(review))
    rating_labels = raw_dataframe['reviews.rating'] # original ratings
    return rating_text, rating_labels
    