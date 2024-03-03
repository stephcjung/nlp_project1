import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    #sequence is a list of strings
    #n is the number of strings per gram
    #return a list of tuples
    
    seq = sequence.copy()
    for j in range(1,n):
        seq.insert(0,"START")
    if (n==1):
        seq.insert(0,"START")
        
    seq.append("STOP")
    #print(seq)
    final_list = []
    
    
    for i in range(len(seq) - n + 1):
        lst = []
        for k in range(n):
            lst.append(seq[i+k])
        tup = tuple(lst)
        #print(tup)
        final_list.append(tup)


    return final_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
        
        # Declare counters
        self.num_sentences = 0
        self.num_tokens = 0     #Used as denominator in unigram raw prob (excludes "START")
        self.lexicon_size = -1     #This is |V|; number of different words
        #starts at -1 to account for 'START' being counted as a word (we want to discount it)
        
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
    
        


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        
        #have to make sure UNK is in there
        self.unigramcounts[('UNK',)] = 1

        #unigrams:
        for sentence in corpus:
            self.num_sentences += 1
            
            ngrams_sentence_uni = get_ngrams(sentence, 1)
            ngrams_sentence_bi = get_ngrams(sentence, 2)
            ngrams_sentence_tri = get_ngrams(sentence, 3)
            
            for i in range(len(ngrams_sentence_uni)):
                if (tuple(ngrams_sentence_uni[i]) != ('START',)):
                    self.num_tokens += 1
                if (tuple(ngrams_sentence_uni[i]) not in self.unigramcounts.keys()):
                    self.unigramcounts[tuple(ngrams_sentence_uni[i])] = 1
                    self.lexicon_size += 1
                else:
                    self.unigramcounts[tuple(ngrams_sentence_uni[i])] += 1
                    
            for j in range(len(ngrams_sentence_bi)):
                if (tuple(ngrams_sentence_bi[j]) not in self.bigramcounts.keys()):
                    self.bigramcounts[tuple(ngrams_sentence_bi[j])] = 1
                else:
                    self.bigramcounts[tuple(ngrams_sentence_bi[j])] += 1
            
            for k in range(len(ngrams_sentence_tri)):
                if (tuple(ngrams_sentence_tri[k]) not in self.trigramcounts.keys()):
                    self.trigramcounts[tuple(ngrams_sentence_tri[k])] = 1
                else:
                    self.trigramcounts[tuple(ngrams_sentence_tri[k])] += 1

        #print(self.unigramcounts)
        print("\n")
        #print(self.bigramcounts)
        print("\n")
        #print(self.trigramcounts)


        
        return
    
    

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        
        if count(u,v,w) is count('START','START', w):
            P(u,v,w) = count(u,v,w) / num_sentences
        if count(u,v,w) is 0 and count(u,v) is 0: 
            P(u,v,w) is 1/lexicon_size (number of unique words, exlcuding START)
        
        """
        uv = (trigram[0], trigram[1])
        if (trigram not in self.trigramcounts.keys()):
            if (uv not in self.bigramcounts.keys()):
                return  1/(len(self.lexicon)-1)
            return 0.0
        if (uv == ('START','START')):
            return self.trigramcounts[trigram]/self.num_sentences
        #print(trigram)
        #print(uv)

        
        #print(self.trigramcounts[trigram])
        #print(self.bigramcounts[uv])
        return self.trigramcounts[trigram]/self.bigramcounts[uv]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        
        if count(u,v) is 0: 
            count(u) should never be 0 so P(u,v)=0.0
        if count(u,v) is count('START', v):
            P(u,v) = count('START',v) / num.sentences
        """
        u = (bigram[0],)
        if (bigram not in self.bigramcounts.keys()):
            return 0.0
        if (u == ('START',)):
            return self.bigramcounts[bigram]/self.num_sentences
        
        #print(bigram)
        #print(u)

        
        #print(self.bigramcounts[bigram])
        #print(self.unigramcounts[u])
        return self.bigramcounts[bigram]/self.unigramcounts[u]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        
        if unigram is unseen: should never happen, UNK
        if 'START': P(v) = 0.0
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
        #print(unigram)
        #print(self.num_tokens)
        #print(unigram)
        if (unigram == ('START',)):
            return 0.0
       # if (unigram not in self.unigramcounts.keys()):
        #    return 1/self.unigramcounts[('UNK',)] 
        #self.raw_unigram_probability(('UNK',)) #this should never happen
        else:
            #print(self.unigramcounts[unigram])
            return self.unigramcounts[unigram]/self.num_tokens

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        raw_tri = self.raw_trigram_probability(trigram)
        bigram = (trigram[1],trigram[2])
        raw_bi = self.raw_bigram_probability(bigram)
        unigram = (trigram[2],)
        raw_uni = self.raw_unigram_probability(unigram)
        #print(raw_tri)
        #print(raw_bi)
        #print(raw_uni)
        return (lambda1*raw_tri)+(lambda2*raw_bi)+(lambda3*raw_uni)
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigramslst = get_ngrams(sentence, 3)
        tot_logprob = 0.0
        #have to make sure trigrams' probabilities aren't counted multiple times
        #if the trigram appears more than once        
        
        for i in trigramslst:
            smoothprob = self.smoothed_trigram_probability(i)
            #print(smoothprob)
            tot_logprob += math.log2(smoothprob)
        
                
        return float(tot_logprob)
    
    

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        
        #M is the total number of word tokens
        #m is the number of sentences in the text corpus
        
        sum_logprobs = 0.0
        n=0
        
        for sentence in corpus:
            sum_logprobs += self.sentence_logprob(sentence)
            for i in sentence:
                if (i != 'START'):
                    n += 1

        l = sum_logprobs / n
            
        return float(2**(-l)) 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0.0
        correct = 0.0      
 
        ''' 
        We are assuming that the files in testdir1 should have the same category as
        training_file1 to be deemed correct
        We are assuming that the files in testdir2 should have the same category as 
        training_file2 to be deemed correct
        '''
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            # .. 
            total += 1
            if (pp1 < pp2):
                correct += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            # .. 
            total += 1
            if (pp2 < pp1):
                correct += 1
        
        return correct/total

if __name__ == "__main__":
    
    print(sys.argv[1])

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)
    


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    print(acc)

