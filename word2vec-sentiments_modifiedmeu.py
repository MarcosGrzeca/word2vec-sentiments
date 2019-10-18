# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import logging
import sys

#save classifier
# import pickle
from joblib import dump, load

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)


log.info('source load')
train_source = {'meu/meu-train-neg.txt':'TRAIN_NEG', 'meu/meu-train-pos.txt':'TRAIN_POS', 'meu/meu-test-neg.txt':'TEST_NEG', 'meu/meu-test-pos.txt':'TEST_POS', 'meu/meu-train-unsup.txt':'TRAIN_UNS'}
test_source = {'meu/meu-test-neg.txt':'TEST_NEG', 'meu/meu-test-pos.txt':'TEST_POS'}

log.info('TaggedDocument')
train_sentences = TaggedLineSentence(train_source)
test_sentences = TaggedLineSentence(test_source)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, vector_size=150, sample=1e-4, negative=5, workers=50,epochs=20)
model.build_vocab(train_sentences.to_array())

log.info('Epoch')

# log.info('EPOCH: {}'.format(epoch))
model.train(train_sentences.sentences_perm(),total_examples=model.corpus_count,epochs=model.epochs)

log.info('Model Save')
model.save('./rezando.d2v')
model = Doc2Vec.load('./rezando.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((1624, 150))
train_labels = numpy.zeros(1624)

print(model.most_similar('good'))

for i in range(1172):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for index in range(452):
    prefix_train_neg = 'TRAIN_NEG_' + str(index)
    train_arrays[i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 0
    i = i +1

test_arrays = numpy.zeros((406, 150))
test_labels = numpy.zeros(406)



i = 0
# for index, i in enumerate(test_sentences):
for i in range(293):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for index in range(113):
    prefix_test_pos = 'TEST_NEG_' + str(index)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 0
    i = i +1

log.info('Fitting')
classifier = LinearSVC()
classifier.fit(train_arrays, train_labels)

#numpy.savetxt("kagggle_train_LogisticRegression_labels.csv", numpy.asarray(train_labels), delimiter=",")
#numpy.savetxt("kagggle_teste_LogisticRegression_labels.csv", numpy.asarray(test_labels), delimiter=",")

# classifier = LogisticRegression()
# classifier.fit(train_arrays, train_labels)

# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

final = classifier.predict(test_arrays)

log.info(classifier.score(test_arrays, test_labels))
dump(classifier, 'classifier.joblib')

print(precision_recall_fscore_support(final, test_labels, average='micro'))
print(precision_recall_fscore_support(final, test_labels, average='micro'))

print(f1_score(final, test_labels, average="macro"))

print(f1_score(final, test_labels, average="micro"))

print(f1_score(final, test_labels, average="weighted"))
print(precision_score(final, test_labels, average="macro"))
print(recall_score(final, test_labels, average="macro"))   

#classifier = load('classifier.joblib') 

matrix = confusion_matrix(final, test_labels)
print(matrix)

#https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/