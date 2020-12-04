import argparse # to parse command line args
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from joblib import dump, load # for saving model

import warnings
warnings.filterwarnings("ignore")

from data import get_sentiment_corpus_and_labels
from data import get_amazon_reviews_corpus_and_labels

def get_args():
    parser = argparse.ArgumentParser(description="Choose model.")

    parser.add_argument("--model", type=str, choices=['logistic', 'svm'], required=True, help="Which model do you want to use (logistic, svm, etc.) .")
    # parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
    #                     help="Operating mode: train or test.")
    parser.add_argument("--kernel", type=str, required=True, choices=["count", "tfidf"],
                        help="Kernel function to use. Options include: count or tfidf.")
    # parser.add_argument("--model-file", type=str, required=True,
    #                     help="The name of the model file to create (for training) or load (for testing).")
    # parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")
    args = parser.parse_args()

    return args

def baseline_classify():
    sentiment_corpus, sentiment_labels = get_sentiment_corpus_and_labels()
    sentiment_corpus = np.array(sentiment_corpus)
    sentiment_labels = np.array(sentiment_labels)
    X_train, X_test, Y_train, Y_test = train_test_split(sentiment_corpus, sentiment_labels, test_size=0.3, random_state=10)

    # this section adapted from https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44
    if args.kernel == 'count':
        cv = CountVectorizer(binary=True, analyzer = 'word', min_df = 10, max_df = 0.95) # creates matrix of counts drops words in less than 10 docs or more than 95 percent of docs
    elif args.kernel == 'tfidf':
        cv = TfidfVectorizer()
    cv.fit_transform(X_train) # returns document-term matrix 
    train_feature_set=cv.transform(X_train) # returns document-term matrix (represents frequency of terms in strings, rows are strings cols are terms)
    test_feature_set=cv.transform(X_test) # type is scipy.sparse.csr.csr_matrix
    # build the appropriate model
    if args.model == "logistic":
        lr = LogisticRegression(solver = 'liblinear', random_state = 42, max_iter=1000) # define classifier lr is simple for baseline
        lr.fit(train_feature_set, Y_train) # fit model
        y_pred = lr.predict(test_feature_set) # make predictions
        print("Accuracy: ", round(accuracy_score(Y_test,y_pred),7)) # get accuracy to 7 decimal places
        report = classification_report(Y_test, y_pred, output_dict=True)
        print('positive: ', report['1'])
        print('negative: ', report['0'])

        # set hyperparameters tuning grid 
        param_grid={'solver':['newton-cg', 'lbfgs','liblinear', 'saga'],
        'tol':[1e-5, 1e-4,1e-3, 1e-2, 1e-1], 'C': [0.25, 0.5, 1.0],
        'max_iter': [i for i in range(10,5000,20)]}
        model = LogisticRegression()

    elif args.model == "svm":
        # Perform classification with SVM, kernel=linear
        classifier_linear = svm.SVC(kernel='linear')
        classifier_linear.fit(train_feature_set, Y_train)
        y_pred = classifier_linear.predict(test_feature_set)
        print("Accuracy: ", round(accuracy_score(Y_test,y_pred),7))
        report = classification_report(Y_test, y_pred, output_dict=True)
        print('positive: ', report['1'])
        print('negative: ', report['0'])

        # set hyperparameters tuning grid 
        param_grid={'kernel': ['linear', 'rbf', 'poly','sigmoid'], 'gamma': ['auto', 1e-3, 100,10,'scale'],
        'tol':[0.0001,0.001,0.01], 'C':[0.1,1,10,100],
        'degree':[i for i in range(1,500,2)], 
        'max_iter': [i for i in range(10,5000,20)], 'probability':[True]}
        model = svm.SVC()

    # Perform a hyperparameter search using training data
    hp_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, random_state=12345, cv=5)
    # hp_search = BaggingClassifier(base_estimator=hp_search, n_estimators=10, random_state=0)
    #hp_search = GridSearchCV(estimator=model, param_grid=param_grid)
    hp_search = hp_search.fit(train_feature_set, Y_train)
    predictions = hp_search.predict(test_feature_set)
    dump(hp_search, 'models\svm.joblib') # save model
    #print('tuned parameters: {}'.format(hp_search.best_params_))
    print('tuned parameters: {}'.format(hp_search.get_params()))
    #print('best score is {}'.format(hp_search.best_score_))
    print('best score is {}'.format(hp_search.score(test_feature_set, Y_test)))

    return hp_search



if __name__ == "__main__":
    args = get_args()
    baseline_classify()

# Just Ideas: we want to train the model, save the model, and then predict on the amazon reviews.
# each amazon review is 1+ sentences long. For each sentence in each review, we want a 0 or 1. 
# for each review, we sum these 0s and 1s to get a score that we divide by number of sentences in the review
# we use this new score (sum / num sentences) to assign new rating to that review
# save these ratings and compare against review 'labels' (original ratings)

# how to save model?
# how to predict on sentences one at a time?