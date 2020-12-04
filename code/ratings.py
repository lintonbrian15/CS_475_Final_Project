from joblib import dump, load
from data import get_amazon_reviews_corpus_and_labels
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

if __name__ == "__main__":
    # just trying to load model
    svm = load('models\svm.joblib')
    cv = pickle.load(open("tfidf.pickle", "rb"))
    amazon_corpus, amazon_labels = get_amazon_reviews_corpus_and_labels()
    test = cv.transform(amazon_corpus[4])
    print(amazon_corpus[4])
    #test = cv.transform(amazon_corpus[0[0]])
    predictions = svm.predict(test)
    print(predictions)
