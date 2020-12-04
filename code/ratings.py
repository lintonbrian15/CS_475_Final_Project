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
    predictions = []
    for review in amazon_corpus: # get sentiment of each sentence in each review
        test = cv.transform(review)
        prediction = svm.predict(test)
        predictions.append(prediction)
    new_ratings = []
    for prediction in predictions: # generate new ratings
        prediction = prediction.astype(np.float)
        pred_sum = prediction.sum() # number of 1s i.e. number of positive sentences in the review
        pred_len = len(prediction)
        pred_ratio = pred_sum / pred_len # get ratio of positive sentences to total sentences
        # generate new rating
        #[0.00-0.19] 1 star, [0.20-0.39] 2 stars, [0.40-0.59] 3 stars, [0.60-0.79] 4 stars,
        #[0.80-1.00] 5 stars
        if pred_ratio >= 0.80:
            new_ratings.append(5)
        elif pred_ratio >= 0.60:
            new_ratings.append(4)
        elif pred_ratio >= 0.40:
            new_ratings.append(3)
        elif pred_ratio >= 0.20:
            new_ratings.append(2)
        elif pred_ratio >= 0.00:
            new_ratings.append(1)
    #test = cv.transform(amazon_corpus[0[0]])
