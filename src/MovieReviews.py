import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def deal_with_word(review):     # only keep the English words
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower()
    return words

def main():
    data_dir = '../input/MovieReview'
    train = pd.read_csv(data_dir + '/labeledTrainData.tsv', delimiter="\t")
    test = pd.read_csv(data_dir + '/testData.tsv', delimiter="\t")

    #print(train.head())
    #print(train.shape)
    #print(test.shape)

    y_train = train['sentiment']

    train_data = []
    for i in range(0,len(train['review'])):
        if i % 4000 == 0:
            print ('training process line: ', str(i))
        train_data.append(deal_with_word(train['review'][i]))
    train_data = np.array(train_data)

    test_data = []
    for i in range(0,len(test['review'])):
        if i % 4000 == 0:
            print ('testing process line: ', str(i))
        test_data.append(deal_with_word(test['review'][i])) 
    test_data = np.array(test_data)    

    vectorizer = CountVectorizer()

    data_train_count = vectorizer.fit_transform(train_data)   # fit -- count the word,    transform -- word to vector
    data_test_count = vectorizer.transform(test_data)

    tfidf = TfidfVectorizer(
           ngram_range=(1, 3),
           use_idf=1,
           smooth_idf=1,
           stop_words = 'english') 
    
    data_train_count_tf = tfidf.fit_transform(train_data)
    data_test_count_tf  = vectorizer.transform(test_data)


    clf = MultinomialNB()
    clf.fit(data_train_count, y_train)
    
    print ("cross-validation score: ", np.mean(cross_val_score(clf, data_train_count, y_train, cv=10, scoring='accuracy')))
    print ("cross-validation score: ", np.mean(cross_val_score(clf, data_train_count_tf, y_train, cv=10, scoring='accuracy')))

    predict = clf.predict(data_test_count)

    df = pd.DataFrame({'id': test['id'], 'sentiment': predict})
    df.to_csv('submission.csv', index = False, header = True)

if __name__ == '__main__':
    main()
