from joblib import dump, load
from data import get_amazon_reviews_corpus_and_labels
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

if __name__ == "__main__":
    # just trying to load model
    svm = load('models\svm.joblib')
    amazon_corpus, amazon_labels = get_amazon_reviews_corpus_and_labels()
    cv = TfidfVectorizer()
    test = [amazon_corpus[0][0]]
    train = ["yes 's shiny front side love"]
    cv.fit_transform(train)
    test_set = cv.transform(test)
    print(test_set)
    predictions = svm.predict(test_set)
    print(predictions)
