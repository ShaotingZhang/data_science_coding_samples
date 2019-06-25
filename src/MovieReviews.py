import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def deal_with_word(review):     # only keep the English words
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower()
    return(words)

def main():
    data_dir = '../input/MovieReview'
    train = pd.read_csv(data_dir + '/labeledTrainData.tsv', delimiter="\t")
    test = pd.read_csv(data_dir + '/testData.tsv', delimiter="\t")


    #print(train.head())
    #print(train.shape)
    #print(test.shape)


    y_train = train['sentiment']

    train_data = []
    for reviews in train['review']:
        train_data.append(deal_with_word(reviews))
    train_data = np.array(train_data)

    test_data = []
    for reviews in test['review']:
        test_data.append(deal_with_word(reviews))
    test_data = np.array(test_data)


    tfidf = TfidfVectorizer(
                ngram_range =(1, 2),
                use_idf = 1,
                smooth_idf = 1,
                stop_words = 'english')    # remove English stop words to improve the efficiency

    vectorizer = CountVectorizer()
    train_data_count = vectorizer.fit_transform(train_data)     # fit -- count the word,    transform -- word to vector
    test_data_count = vectorizer.transform(test_data)

    word_freq_df = pd.DataFrame({'term': tfidf.get_feature_names(), 'tfidf':data_train_count_tf.toarray().sum(axis=0)})
    plt.plot(word_freq_df.occurrences)

    plt.show()

    word_freq_df_sort = word_freq_df.sort_values(by=['tfidf'], ascending=False)
    word_freq_df_sort.head()

    clf = MultinomialNB()
    clf.fit(train_data_count, y_train)
    predict = clf.predict(test_data_count)

    test = clf.predict(train_data_count)

    df = pd.DataFrame({'id': test['id'], 'sentiment': predict})
    df.to_csv('submission.csv', index = False, header = True)

if __name__ == '__main__':
    main()
