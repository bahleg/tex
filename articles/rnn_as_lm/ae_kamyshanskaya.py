"""
Autoencoder with tight weights, sigmoid in encoder layer, linear in decoder layer
"""
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
import theano
import theano.tensor as T
import numpy as np
import random
import os
from theano.tensor.shared_randomstreams import RandomStreams
from laplace import  get_log_constant
srng = RandomStreams(seed=234)
data = np.load('brown.npy')
input_dim = 150
ae_dim = 50 #hidden dimmension
lr = 0.001
batch_size = 100
sigma = 0.0001
"""
model
"""
X = T.matrix()
rv_n = srng.normal((X.shape))
params = theano.shared(np.random.randn(input_dim*ae_dim))
We = params.reshape((input_dim,ae_dim))
Wd = We.T
encoded = T.nnet.sigmoid(T.dot(X+rv_n*sigma, We)) #training as denoising encoder, because denosing is equal to contractive, but simplier in training
decoded = T.dot(encoded, Wd)
"""
training
"""
if os.path.exists('ae_params.npy'):
	params.set_value(np.load('ae_params.npy'))
else:
	cost = T.sum(T.sum((X - decoded)**2, axis=1))
	grad = T.grad(cost, wrt=params)
	train = theano.function([X], cost, updates=[(params, params - grad*lr)])
	elems = range(data.shape[0])
	for i in xrange(0, 200):
		random.shuffle(elems)
		print train(data[elems[:batch_size]])
	np.save('ae_params',params.eval())
"""
getting normalized proba. We must find x, that has the maximum proba for Laplace appr.
"""
x0 = theano.shared(np.random.randn(150)) 
unnormalized_log_proba = T.sum(T.maximum(0,T.dot(x0,We)))- T.dot(x0,x0)#T.maximum(0, x) is a approximation of T.log(1+T.exp(-x)) from Kamyshanskaya's article
grad2 = T.grad(unnormalized_log_proba, wrt=x0)
lr2 = 0.0001
train2 = theano.function([],unnormalized_log_proba, updates = [(x0, x0+lr2*grad2) ])
logZ = get_log_constant(unnormalized_log_proba, x0) #function from laplace appr. code
logZ_function = theano.function( [],logZ)
#note, since the hessian can be semi-definite in maximum point, the constant can be NaN, so we try to start approximation few times 
found_const = False
found_const_iter = 0

while not found_const:
	best_peak = None
	best_proba = -9999
	found_const_iter+=1
	for i in xrange(0, 100000):
		current_proba = train2()
		if current_proba>best_proba:
			best_peak = x0.eval()
			best_proba = current_proba
			print 'peak find', best_proba, current_proba, found_const_iter
			x0.set_value(best_peak)
	result = logZ_function()#try to get const
	if np.isnan(result):
		print 'NOT FOUND'
		x0.set_value(np.random.randn(150))
	else:
		print result#nb: can be > 1 since continuous
		found_const = True	

