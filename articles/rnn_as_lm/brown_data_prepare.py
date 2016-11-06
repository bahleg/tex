from gensim.models.word2vec import Word2Vec
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
import theano
import theano.tensor as T
import numpy as np
w2v_path = 'brown.w2v'
corpora_path = 'brown.txt' #of course it's an overfit
ngram_order = 3
tokenizer = WordPunctTokenizer()
sent_tok = PunktSentenceTokenizer()
w2v_model = Word2Vec.load(w2v_path)
data = []#for autoencoder it's a matrix, not tensor-3
centroid = np.mean(w2v_model.syn0, axis=0)
#using sigmoid, so we need to normalize vectors
min_w2v = np.min(w2v_model.syn0, axis=0)
max_w2v = np.max(w2v_model.syn0, axis=0)
print 'loading data'
with open(corpora_path) as inp:
	text = inp.read()
sentences = sent_tok.tokenize(text)

print 'vectorizing data'
for sent in sentences:
	tokens = tokenizer.tokenize(sent)
	
	for i in xrange(0, len(tokens)-ngram_order):
		ngram_slice = tokens[i:i+ngram_order]
		ngram = [] 
		for t in ngram_slice:
			try:
				ngram.extend((w2v_model[t]-min_w2v)/max_w2v)
			except KeyError:
				ngram.extend((centroid-min_w2v)/max_w2v)		
		data.append(np.array(ngram))

print 'total n-gram count:', len(data)
np.save('brown',data)
