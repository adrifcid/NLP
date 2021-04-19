import scipy as sp
import numpy as np
import pickle#cPickle
import re
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
#from ipynb.fs.full.FINAL_train_models import edit_dist

class MyTfidfVectoriser:
    
    """
    Hand-coded class to compute the tf-idf vectorisation of a list of documents based on 
    06_documents_representation_and_retrieval.ipynb. The main methods are fit() and transform(), 
    to be used in that order, although auxiliary functions like build_vocabulary() are also 
    practical for independent use. I add the methods save() and load() to be able to store the
    attributes (__dict__) of an instance after fit, as it takes very long (about 45 min for 
    roughly 1e6 documents). The vectorisation results and syntax emulate those of the
    corresponding sklearn method.
    
    NB: I kept having problems with sp.sparse.lil_matrix() for getting the same data type as sklearn 
    ("float64") when #rows>1 (I tried specifying the dtype param and also imposing it afterwards
    with the method astype(), both to no avail), so I only use it to construct the matrices (as
    it is said to be more efficient for that than sp.sparse.csr_matrix, and using np.arrays requires
    too much memory) and convert them to csr_matrix afterwards (better for slicing and mathematical 
    operations).
    """
    
    #def __init__(self, corpus, splitter):
    #    self.corpus = corpus
    #    self.splitter = splitter
    
    def build_vocabulary(self, corpus, splitter_pattern='(?u)\\b\\w\\w+\\b'):
        """
        Takes a corpus (list of documents in string format) and splitter pattern (regex pattern 
        to specify what is considered a token).
       
        Returns:
        vocabulary: a vocabulary of the corpus.
        X_w: a dict with the number of documents in the corpus containing at least one instance 
        of  a given key/token.
        word2ind: dictionary indexing tokens as they appear in vocabulary.
        ind2word: same as above, but reversed.
        word_count: #tokens in the corpus.
        """
        splitter = re.compile(splitter_pattern)
        vocabulary = set()
        X_w = dict()   # number of documents that contain word w (any nonzero number of times)
        for document in corpus:
            words = set(splitter.findall(document.lower()))
            # fill up vocabulary 
            vocabulary = vocabulary.union(words)
            # fill up X_w
            for w in words:
                X_w[w] = X_w.get(w, 0) + 1
        word2ind = {v:i for i,v in enumerate(vocabulary)}
        ind2word = {v:k for k,v in word2ind.items()} 
        word_count = len(word2ind)
            
        return vocabulary, X_w, word2ind, ind2word, word_count


    def term_freq(self, documents, normalize=False):
        
        """
        Takes a list of documents in string format (has to be a list even for one document
        or len(documents) will return its number of characters) and a flag to decide whether to
        normalise tf matrix or not (defaults to False because the tfidf is already normalised 
        by rows).
       
        Returns:
        tf: term-frequency matrix (each row is the BoW vector of a document).
        """
        n_documents = len(documents)
        word2ind = self.word2ind
        splitter = self.splitter
        n_features = len(word2ind)
        # term-frequency matrix (# docs x # words)
        # lil_matrix is more efficient.for changing matrix structure
        tf = sp.sparse.lil_matrix( (n_documents, n_features), dtype=float)
        #tf = np.zeros([n_documents, n_features])
        #tf = tf.astype('float64')
        #print(tf.data.dtype)
        
        n_oov_words = 0
        # fill up tf
        for idx, doc in enumerate(documents):
            words = splitter.findall(doc.lower())
            for w in words:
                try:
                    tf[idx, word2ind[w]] += 1
                except:
                    n_oov_words += 1
                    print(f"word '{w}' is out of vocabulary")
                    
            # normalisation defaults to False because the final tfidf is always normalised
            if normalize:
                tf[idx, :] = tf[idx, :].multiply(1/sp.sparse.linalg.norm(tf[idx, :]))
                #tf[idx, :] /= np.linalg.norm(tf[idx, :])
        #tf = tf.astype('float64')
        tf = tf.tocsr()
        #print(tf.data.dtype)
        self.n_oov_words = n_oov_words
        
        return tf
        
    def inv_doc_freq(self, X_w, word2ind, corpus_n_documents, sklearn=True):
        
        """
        Takes dictionary with the counts of each token in a corpus (X_w), another dict indexing 
        such tokens, the number of docs of such a corpus and a flag specifying whether to use 
        sklearn's defition of the inverse-document-frequency (idf) vector or the one seen in 
        class (sklearn=False; defaults to True for checks). 
       
        Returns:
        idf: inverse-document-frequency vector.
        """

        n_features = len(word2ind)
        #idf = np.zeros([1, n_features])
        idf = sp.sparse.lil_matrix( (1, n_features), dtype=float)
        
        # if sklearn = True, use its definition. If not, use the one seen in class 
        #have to use this because idf += 1 is not supported with sparse matrices
        if sklearn==True:
            add = 1
        else:
            add = 0
        for w in X_w:
            # fill up idf
            idf[0, word2ind[w]] = np.log(corpus_n_documents/(1 + X_w[w])) + add

        idf = idf.tocsr()
        return idf
    
    def fit(self, corpus, splitter_pattern='(?u)\\b\\w\\w+\\b', sklearn_idf=True): 
        """
        Fit method of the vectoriser.
        
        Takes a corpus (list of docs in string format; has to be a least even for just 1 document, 
        cf. above), a splitter pattern (to tokenise docs) and an idf flag to know which definition of
        the inverse-document-frequency vector to use (sklearn's or the one seen in class).
       
        Returns:
        Message confirming fit has been performed, allowing to call transform() next.
        """
        corpus_n_documents = len(corpus)
        splitter = re.compile('(?u)\\b\\w\\w+\\b')
        self.splitter = splitter
        self.vocabulary, self.X_w, self.word2ind, self.ind2word, self.word_count = self.build_vocabulary(corpus, splitter_pattern=splitter_pattern)
        self.idf = self.inv_doc_freq(self.X_w, self.word2ind, corpus_n_documents, sklearn=sklearn_idf)

        print(f"Corpus fitted: there are {self.word_count} words.")
        
    def transform(self, documents, normalize_tf=False):
        
        """
        Transform method of the vectoriser.
        
        Takes a list of documents to vectorise (in string format) and a flag specifying whether 
        or not to normalise tf matrix (generally redundant). Documents have to be in a list even
        for just one (cf. above).
        
        Returns:
        tfidf: matrix of vectorised documents.
        """
        # ojo, need to pass input documents as a list, otherwise if only one 
        # document this will give its length, not 1
        n_documents = len(documents)
        self.tf = self.term_freq(documents, normalize=normalize_tf)
        self.tfidf = sp.sparse.lil_matrix( (n_documents, self.word_count), dtype="float64")
        #self.tfidf = sp.sparse.csr_matrix( (n_documents, self.word_count), dtype=float)

        for row_idx in range(n_documents):
            self.tfidf[row_idx, :] = self.tf[row_idx,:].multiply(self.idf)
            #self.tfidf[row_idx, :] *= self.idf
            self.tfidf[row_idx, :] /= sp.sparse.linalg.norm(self.tfidf[row_idx, :])
            #self.tfidf[row_idx, :] /= np.linalg.norm(self.tfidf[row_idx, :])
            
        self.tfidf = sp.sparse.csr_matrix(self.tfidf)
        #print(self.tfidf[0].sum(), self.tfidf[1].sum())
        return self.tfidf

    def save(self, filename):
        """Save class in a file with path 'filename'."""
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)        

    def load(self, filename):
        """Load class from a file with path 'filename'."""
        with open(filename, 'rb') as file:
            self.__dict__ = pickle.load(file)      
            

def cast_list_as_strings(mylist):
    """
    Returns a list as a list of strings
    """
    #assert isinstance(mylist, list), f"the input mylist should be a list, and it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def euclid_dist(x,y):
    """Returns Euclidean distance of two sparse vectors. I found this to be faster than sklearn.metrics.pairwise.euclidean_distances"""
    # substraction is more efficient in lil format, but changing it for every pair of rows takes longer
    #x = x.tolil(); y = y.tolil()
    d = x-y
    return sp.sparse.linalg.norm(d)


def get_features_from_df(df, vectoriser, add_dist=False, dist=cos_sim):
    """
    Returns a sparse matrix containing the features built by the input vectoriser with the 
    option of adding a feature among cosine similarity and Euclidean distance.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    X_q1 = vectoriser.transform(q1_casted)
    X_q2 = vectoriser.transform(q2_casted)    
    X_q1q2 = sp.sparse.hstack((X_q1,X_q2))
    #print(X_q1q2.shape)
    
    if add_dist==True:
        n_samples = df.shape[0]
        # lil format for efficient sparse-strucure modifications
        feature = sp.sparse.lil_matrix( (n_samples, 1), dtype=float)

        if dist==euclid_dist:  #Euclidean distance
            # matrices are in csr format, convenient for efficient row slicing
            for i in range(n_samples):
                feature[i,0] = dist(X_q1[i,:], X_q2[i,:])   
                # normalise
                feature = feature.tocsr()
                feature /= sp.sparse.csr_matrix.max(feature)
        elif dist==cos_sim:   # cosine similarity (same info as cosine distance with less computation)
            # matrices are in csr format, convenient for efficient row slicing
            for i in range(n_samples):
                feature[i,0] = dist(X_q1[i,:], X_q2[i,:])[0,0] 
        #elif dist==edit_dist:  #edit distance
        #    for i in range(n_samples):
        #        feature[i,0] = dist(q1_casted[i], q2_casted[i])
        #        # normalise
        #        feature = feature.tocsr()
        #        feature /= sp.sparse.linalg.norm(feature)  
        else:
            print("Only valid values of dist are euclid_dist and cos_sim. Please enter one of them.")
        X_q1q2 = sp.sparse.hstack((X_q1q2,feature))  
        #print(X_q1q2.shape)
            
    return X_q1q2


def get_mistakes(clf, X_q1q2, y, neural_net=False):

    """Returns indices of erroneus predictions, as well as the predictions themselves."""
    if neural_net==False:
        predictions = clf.predict(X_q1q2)
    else:
        predictions = clf.predict(X_q1q2)[:,1]
        predictions = np.where(predictions>0.5, 1, 0)
    incorrect_predictions = predictions != y
    incorrect_indices,  = np.where(incorrect_predictions)

    total_mistakes = np.sum(incorrect_predictions)
    error_rate = total_mistakes/predictions.shape[0]
    if total_mistakes==0:
        print("no mistakes in this df")
    else:
        print(f"error rate: {error_rate} \n total mistakes: {total_mistakes}")
        return incorrect_indices, predictions


def print_mistake_k(df, k, mistake_indices, predictions):
    """For a given mistaken prediction, this prints the pair of questions, the ground truth and the prediction. """
    print(df.iloc[mistake_indices[k]].question1)
    print(df.iloc[mistake_indices[k]].question2)
    print("true class:", df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])


def print_auc_logloss(clf, X, y):
    """Prints the AUC of the predicted probabilities of a given classifier. Also prints the log loss, for comparison with Kaggle results (as they use the latter metric)."""
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    logloss = log_loss(y, probs)
    print(f"AUC: {auc}")
    print(f"log loss: {logloss}")

def convert_sparse_matrix_to_ordered_sparse_tensor(X):
    """Converts input sparse matrix to the format required by tensorflow."""
    indices = np.mat([X.row, X.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, X.data, X.shape))

