"""
toy example for word2vec
"""
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
tokenizer = WordPunctTokenizer()
sent_tok = PunktSentenceTokenizer()
with open('brown.txt') as inp:
	data = inp.read()
sentences = sent_tok.tokenize(data)
for i in xrange(0, len(sentences)):
	sentences[i] = tokenizer.tokenize(sentences[i].lower())

model = Word2Vec(sentences, size=50, iter=3)
model.save('brown.w2v')

