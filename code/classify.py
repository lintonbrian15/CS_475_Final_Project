import argparse # to parse command line args

from data import get_sentiment_corpus_and_labels
from data import get_amazon_reviews_corpus_and_labels

def get_args():
    parser = argparse.ArgumentParser(description="Choose model.")

    parser.add_argument("--model", type=str, required=True, help="Which model do you want to use (baseline, svm, etc.) .")

    args = parser.parse_args()

    return args

def baseline_classify():
    sentiment_corpus, sentiment_labels = get_sentiment_corpus_and_labels()
    print(type(sentiment_corpus))
    print(type(sentiment_labels))
    print(sentiment_corpus[:3])
    print(sentiment_labels[:3])

if __name__ == "__main__":
    args = get_args()
    baseline_classify()