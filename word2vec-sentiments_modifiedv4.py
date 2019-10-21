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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn import svm

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
train_source = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-unsup.txt':'TRAIN_UNS'}
# train_source = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
test_source = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}

log.info('TaggedDocument')
train_sentences = TaggedLineSentence(train_source)
test_sentences = TaggedLineSentence(test_source)

model = Doc2Vec.load('./imdbcv2.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((25000, 150))
train_labels = numpy.zeros(25000)

print(model.most_similar('good'))

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

log.info(train_labels)

test_arrays = numpy.zeros((25000, 150))
test_labels = numpy.zeros(25000)

i = 0
# for index, i in enumerate(test_sentences):
for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    # feature = model.infer_vector(i[0])

    # if index <12500:
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0


log.info('Fitting')
#classifier = LinearSVC(max_iter = 2000)
#classifier = LinearSVC(max_iter = 2000, kernel = "poly")
#classifier = svm.SVC(max_iter = 2000, kernel = "poly")
classifier = svm.SVC(kernel = "poly")

#numpy.savetxt("kagggle_train_LogisticRegression_labels.csv", numpy.asarray(train_labels), delimiter=",")
#numpy.savetxt("kagggle_teste_LogisticRegression_labels.csv", numpy.asarray(test_labels), delimiter=",")

# classifier = LogisticRegression()
# classifier.fit(train_arrays, train_labels)

# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

scoring = {
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'prec_macro': 'precision_macro',
    'rec_macro': 'recall_macro'
}

#scores = cross_val_score(classifier, train_arrays, train_labels, cv=5, scoring=scoring, return_train_score=true)
scores = cross_validate(classifier, train_arrays, train_labels, cv=5, scoring=scoring)

print(scores.keys())
print(scores['test_f1_macro'])
print(scores['test_f1_micro'])