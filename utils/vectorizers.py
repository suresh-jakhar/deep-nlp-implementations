"""
Text vectorization utilities.
"""

import numpy as np
from collections import Counter


class BagOfWords:
    """
    Bag of Words vectorizer implementation.
    """
    
    def __init__(self):
        self.vocabulary = {}
        
    def fit(self, documents):
        """
        Build vocabulary from documents.
        
        Args:
            documents (list): List of text documents
        """
        all_words = set()
        for doc in documents:
            words = doc.split()
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
    def transform(self, documents):
        """
        Transform documents to BoW vectors.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            np.ndarray: BoW matrix
        """
        vectors = []
        for doc in documents:
            vector = np.zeros(len(self.vocabulary))
            word_counts = Counter(doc.split())
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    vector[self.vocabulary[word]] = count
                    
            vectors.append(vector)
            
        return np.array(vectors)


class TFIDF:
    """
    TF-IDF vectorizer implementation.
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        
    def fit(self, documents):
        """
        Build vocabulary and calculate IDF from documents.
        
        Args:
            documents (list): List of text documents
        """
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.split()
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        # Calculate IDF
        n_documents = len(documents)
        word_doc_count = Counter()
        
        for doc in documents:
            words = set(doc.split())
            word_doc_count.update(words)
        
        for word in self.vocabulary:
            self.idf[word] = np.log(n_documents / (word_doc_count[word] + 1))
        
    def transform(self, documents):
        """
        Transform documents to TF-IDF vectors.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        vectors = []
        for doc in documents:
            vector = np.zeros(len(self.vocabulary))
            words = doc.split()
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    tf = count / len(words)
                    tfidf = tf * self.idf.get(word, 0)
                    vector[self.vocabulary[word]] = tfidf
                    
            vectors.append(vector)
            
        return np.array(vectors)
