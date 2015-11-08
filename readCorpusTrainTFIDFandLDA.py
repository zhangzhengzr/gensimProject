#First input parameter is the path of the text(it should be dictionary later) 
#Second input parameter is the path of the corpus iterator
import logging, bz2, gensim, sys
import os.path
from gensim import corpora, models, similarities
import numpy as np
import matplotlib.pyplot as plt
import scipy

logging.basicConfig(filename='wikiTest.log',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	id2word = gensim.corpora.Dictionary.load_from_text(sys.argv[1])
	#dictionary = corpora.Dictionary.load(sys.argv[1])
	corpus = gensim.corpora.MmCorpus(sys.argv[2])
	print (corpus)
	#Train TFIDF Model
	tfidf = models.TfidfModel(corpus, id2word=id2word, normalize=True)
	tfidf.save('tfidfModel.tfidf_model')
	#Train TFIDF corpus
	corpus_tfidf = tfidf[corpus]
	#Using LSI Model to train the topics 
	#lsi = gensim.models.lsimodel.LsiModel(corpus=corpus_tfidf,id2word=id2word,num_topics=100)
	#lsi.show_topics(10,100)
	#lsi.save('lsiModel.lsi')
	#Using LDA Model to train the topics
	lda = gensim.models.ldamodel.LdaModel(corpus = corpus_tfidf, id2word = id2word, num_topics=100, update_every=1, chunksize=10000, passes = 1)
	lda.show_topics(10,100)
	lda.save('ldaModel.lda')
	#lda.save(sys.argv[3])

#Define Kullback-Leibler divergence function
def sym_kl(p,q):
	return np.sum([scipy.stats.entropy(p,q),scipy.stats.entropy(q,p)])

#l=np.array([sum(cont for _, cnt in doc) for doc in corpus])

def arun(corpus,dictionary,min_topics=1,max_topics=150,step=1):
	kl= []
	for i in range(min_topics,max_topics,step):
		lda = models.ldamodel.LdaModel(corpus=corpus, id2word = dictionary, num_topics=i)
		m1 = lda.expElogbeta
		U, cm1, V = np.linalg.svd(m1)