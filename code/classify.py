import argparse # to parse command line args
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from data import get_sentiment_corpus_and_labels
from data import get_amazon_reviews_corpus_and_labels

def get_args():
    parser = argparse.ArgumentParser(description="Choose model.")

    parser.add_argument("--model", type=str, required=True, help="Which model do you want to use (baseline, svm, etc.) .")

    args = parser.parse_args()

    return args

def baseline_classify():
    sentiment_corpus, sentiment_labels = get_sentiment_corpus_and_labels()
    sentiment_corpus = np.array(sentiment_corpus)
    sentiment_labels = np.array(sentiment_labels)
    X_train, X_test, Y_train, Y_test = train_test_split(sentiment_corpus, sentiment_labels, test_size=0.3, random_state=10)
    # this section adapted from https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44
    cv = CountVectorizer(binary=True, analyzer = 'word', min_df = 10, max_df = 0.95) # creates matrix of counts drops words in less than 10 docs or more than 95 percent of docs
    cv.fit_transform(X_train) # returns document-term matrix 
    train_feature_set=cv.transform(X_train) # returns document-term matrix (represents frequency of terms in strings, rows are strings cols are terms)
    test_feature_set=cv.transform(X_test) # type is scipy.sparse.csr.csr_matrix
    lr = LogisticRegression(solver = 'liblinear', random_state = 42, max_iter=1000) # define classifier lr is simple for baseline
    lr.fit(train_feature_set, Y_train) # fit model
    y_pred = lr.predict(test_feature_set) # make predictions
    print("Accuracy: ", round(accuracy_score(Y_test,y_pred),7)) # get accuracy to 7 decimal places


if __name__ == "__main__":
    args = get_args()
    if (args.model == 'baseline'):
        baseline_classify()