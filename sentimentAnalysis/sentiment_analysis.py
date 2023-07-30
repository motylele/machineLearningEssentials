import nltk
import random
from nltk.corpus import movie_reviews


def preprocess_data(num_words=2000):
    """Calculation words frequency distribution and selecting most common words"""
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:num_words]
    doc = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]
    random.shuffle(doc)
    return word_features, doc


def extract_features(doc, word_features):
    """Extracting features from movie reviews"""
    def doc_features(doc):
        doc_words = set(doc)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in doc_words)
        return features
    return [(doc_features(d), c) for (d, c) in doc]


def train_and_evaluate_classifier(train_set, test_set):
    """Training Naive Bayes Classifier and evaluation"""
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    return classifier, accuracy


def display_features(classifier, num_features=5):
    """Displaying most informative features"""
    print("Top {} most informative featuers".format(num_features))
    classifier.show_most_informative_features(num_features)


def main():
    # Data preprocessing
    word_features, doc = preprocess_data()

    # Features Extraction
    featuresets = extract_features(doc, word_features)

    # Feature set creation
    set_size = 150
    train_set, test_set = featuresets[set_size:], featuresets[:set_size]

    # Naive Bayes Classifier training & evaluation
    classifier, accuracy = train_and_evaluate_classifier(train_set, test_set)
    print("Classification accuracy = {}".format(accuracy))

    # Displaying most informative features
    display_features(classifier, 10)

    # Displaying first movie review
    # print(movie_reviews.raw(movie_reviews.fileids()[0]))


if __name__ == "__main__":
    main()
