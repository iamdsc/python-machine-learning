# Implementing Naive Bayes Spam Classifier
import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Reading in the data from dataset
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message':message,'class':classification})
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'message':[],'class':[]})
data = data.append(dataFrameFromDirectory('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/emails/spam','spam'))
data = data.append(dataFrameFromDirectory('C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/emails/ham','ham'))

# Training the classifier
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

# Prediction
examples = ['Free offer for mobile phone discounts', 'Hey, John how are you ?']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)











