# Reading in sentiment data to create corpus

import nltk # Natural Language Toolkit
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# The following was taken from data.py in HW4 from CS_475

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

if __name__ == "__main__":
    amazon_labels = []
    amazon_sentences = []
    with open('datasets/amazon_sentiments.txt') as amazon_sentiments:
        for line in amazon_sentiments:
            line = line.split()
            text = line[:-1] # get all words except label
            string = " "
            amazon_sentences.append(clean_text(string.join(text))) # get cleaned text data
            amazon_labels.append(line[-1]) # get label
          