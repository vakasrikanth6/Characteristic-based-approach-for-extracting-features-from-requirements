# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:55:16 2019

@author: vakas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:10:58 2019

@author: vakas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:33:42 2018

@author: vakas
"""
import pandas as pd
import re
import os
from gensim.models import FastText
from gensim.models import Word2Vec
import numpy as np
import math
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
#from sklearn.metrics import precision_recall_fscore_support
#from prettytable import PrettyTable

nltk.download('averaged_perceptron_tagger')
#corpus= pd.read_excel('BCS_Requirements_english_revision.xlsx', sheetname = "Ground Truth")    #Our Dataset 1
#corpus= pd.read_excel('DigitalHome.xlsx')                                                      #Our Dataset2
corpus= pd.read_excel('BCS_Requirements clustered data.xlsx') 
corpusinitial = corpus['Requirement']
labels_true = corpus['GroundTruth']
InputData = []



#------------------pre processing the document---------------------#

for i in range(len(corpusinitial)):
    check = re.sub('[^a-zA-Z]', ' ', corpusinitial[i])
    check = check.lower()
    check = check.split()
    ps = PorterStemmer()
    check = [ps.stem(word) for word in check if not word in set(stopwords.words('english'))]
    check = ' '.join(check)
    InputData.append(check)
    
#------------------pre processing the document---------------------#
    
  
# Save all sentences as list of lists to train word2vec and FastText models
all_sentences = []

nouns_verbs = []    
#------------------Load only nouns and adjectives from the dataset-----------------#

# nltk.help.upenn_tagset() usage of parts of speech(POS) tagging

for i in range(len(InputData)):

    tokens = re.sub(r"[^a-z0-9]+", " ", InputData[i].lower()).split()
    #all_sentences.append(tokens)    
    
    nouns_verbs_in_nltk = ['NN','NNP','NNPS','NNS','VB','VBD','VBZ'] 
    tags = nltk.pos_tag(tokens)       
    nv_temp = []
    
    for i in range(0,len(tags)-1):
        if tags[i][1] in nouns_verbs_in_nltk:
            nv_temp.append(tags[i][0])
    
    nouns_verbs.append(nv_temp)
    all_sentences.append(nv_temp)
 
#------------------Load only nouns and adjectives from the dataset-----------------#


#-----------------Loading wordembeddings and saving them to a dictionary-------------#
    
#----------------------Glove embeddings--------------------#
#The following method takes Glove Embedding file and stores all words and their embeddings in a dictionary
def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddings[word]=[float(x) for x in vec.strip().split(' ')]
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = "0.0 "*emb_dim
    fe.close()
    
    
#----------------------Glove embeddings--------------------#
    
    
#---------------------Word2vec embeddings with google data-----------#

def loadWord2VecEmbeddings(all_sentences,emb_dim): 
    """EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'    
    print('Indexing word vectors')    
    word2vec = models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True,limit=100000)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))    
    filename = 'word_embedding.txt'
    word2vec.wv.save_word2vec_format(filename, binary= False)"""
    
    f = open(os.path.join('','word_embedding.txt'),encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        Word2vecEmbeddings[word] = coeffs
    Word2vecEmbeddings["zerovec"] = "0.0 "*emb_dim    
    f.close()     
    #print(Word2vecEmbeddings)

"""
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'  
    print('Indexing word vectors') 
    word2vec = models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True,limit)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))    
    for word in list(word2vec.wv.vocab):
        Word2vecEmbeddings[word] = word2vec.wv[word]"""
        
#---------------------Word2vec embeddings with google data-----------#

#----------------Train our own data and give it as input-----------------#
'''
def loadWord2VecEmbeddings(sentences,emb_dim):
    global Word2vecEmbeddings
    model_Word2Vec = Word2Vec(sentences=sentences, size=emb_dim, window=5, min_count=5, workers=4, sg=0)
    for word in list(model_Word2Vec.wv.vocab):
        Word2vecEmbeddings[word] = model_Word2Vec.wv[word]
    Word2vecEmbeddings["zerovec"] = "0.0 "*emb_dim 

print(type(Word2vecEmbeddings['alarm']))
'''
def loadFastTextEmbeddings(sentences,emb_dim):
    global FastTextEmbeddings
    model_FastText = FastText(sentences=sentences, size=emb_dim, window=5, min_count=5, workers=4,sg=1)
    for word in list(model_FastText.wv.vocab):
        FastTextEmbeddings[word] = model_FastText.wv[word]
    FastTextEmbeddings["zerovec"] = "0.0 "*emb_dim
    
#----------------Train our own data and give it as input-----------------#


    
#InputData = InputData.dropna()

#-------Initialize Global variables---------# 
docIDFDict = {}
avgDocLength = 0
#-------Initialize Global variables---------#

#-------------Global variables for Glove Embeddings---------# 

GloveEmbeddings = {}
Word2vecEmbeddings = {}
FastTextEmbeddings = {}
max_query_words = 12
max_passage_words = 50
emb_dim = 300

# Name embddings to be loaded
embeddingFileName = "glove.6B.300d.txt"
loadEmbeddings(embeddingFileName)



#-------------Global variables for Glove Embeddings---------# 

"https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md  download english text file and keep them in same folder as this file"

# Train Word2vec embeddings and create dictionary 
loadWord2VecEmbeddings(all_sentences,emb_dim)
loadFastTextEmbeddings(all_sentences,emb_dim)

#-----------------IDF generator and calculation different variables-----------------#
def IDF_Generator(InputData, delimiter=' ', base=math.e) :
    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for i in range(len(InputData)):
        line = InputData[i]
        doc = line.strip().split(delimiter)
        totalDocLength += len(doc)
        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%10==0):
            print(numOfDocuments)                          

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] +0.5) / (docFrequencyDict[word] + 0.5), base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments
     
    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()

    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)
    
#-----------------IDF generator and calculation different variables-----------------#
    
    
#-------------------------Cosine Similarity Calculations for BM25 ------------------------------#
#The following GetCosineSimilarity will take Query and passage as inputs and outputs cosine similarity between vectors (word embeddings)
#Calclulate cosine similarity with two vectors
def GetCosineSimilarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def GetAverageSentenceVector(sentence,vec_dim,embeddingsType):
    
    avg_sen_vec = np.zeros((vec_dim,), dtype="float32")   
    nwords = 0    
    words = sentence.strip().lower().split()
    
    for word in words:
        if word in embeddingsType.keys():
            nwords = nwords+1
            #word_vec = [float(x) for x in embeddingsType[word].strip().split(' ')]
            word_vec = embeddingsType[word]
            avg_sen_vec = np.add(avg_sen_vec, word_vec)

    if nwords>0:
        avg_sen_vec = np.divide(avg_sen_vec, nwords)
    return avg_sen_vec
    
"Reference link here  https://datascience.stackexchange.com/questions/24855/weighted-sum-of-word-vectors-for-document-similarity"
def GetWeightedSumOfSentenceVector (sentence,vec_dim,embeddingsType):
    avg_sen_vec = np.zeros((vec_dim,), dtype="float32")
    
    nwords = 0   
    words = sentence.strip().lower().split()   
    docTF = {}
    for word in set(words):   #Find Term Frequency of all query unique words
        docTF[word] = words.count(word)
    
    for word in words:
        if word in embeddingsType.keys():
            nwords = nwords+1
            #word_vec = [float(x) for x in embeddingsType[word].strip().split(' ')]
            word_vec = embeddingsType[word]
            weighted_word_vec = [x*docTF[word] for x in word_vec]
            avg_sen_vec = np.add(avg_sen_vec, weighted_word_vec)

    #if nwords>0:
     #   avg_sen_vec = np.divide(avg_sen_vec, nwords)
    return avg_sen_vec
    
def GetAvgOfVectorsSimilarityScore(Query, Passage,embeddingsType):
    
    query_vector = GetAverageSentenceVector(Query,emb_dim,embeddingsType)
    passage_words = GetAverageSentenceVector(Passage,emb_dim,embeddingsType)
    
    score = GetCosineSimilarity(query_vector,passage_words)    
    return score


"Reference link here  https://datascience.stackexchange.com/questions/24855/weighted-sum-of-word-vectors-for-document-similarity"
def GetWeightedSumOfVectorsSimilarityScore(Query, Passage,embeddingsType):
    
    query_vector = GetWeightedSumOfSentenceVector(Query,emb_dim,embeddingsType)
    passage_words = GetWeightedSumOfSentenceVector(Passage,emb_dim,embeddingsType)
    
    score = GetCosineSimilarity(query_vector,passage_words)    
    return score   
    

def GetSemanticSimilarityScore(word, query_words,embeddingsType) :
    
    tmp_score = []    
    if word in embeddingsType.keys():        
        #w_list = [float(x) for x in embeddingsType[word]]
        w_list = embeddingsType[word]        
        #print ("Type w 1 ", type(w_list[1]))
    
        for q_word in query_words:           
            if q_word in embeddingsType.keys():
                #q_w_list = [float(x) for x in embeddingsType[q_word].split(' ')]
                q_w_list = embeddingsType[q_word]
                
            else:
                continue            
            #w = np.array(w_list).reshape(1,-1)            
            #print ("Type w 2 ", type(w))
            #q_w = np.array(q_w_list).reshape(1,-1)           
            #cosine_sim = cosine_similarity(w.reshape(1,-1), q_w.reshape(1,-1))
            cosine_sim = GetCosineSimilarity(w_list,q_w_list)            
            tmp_score.append(cosine_sim)            
            #print ("Cosine Similarity between", word , " and ", q_word , " is ", cosine_sim , " ." )
            
        score = max(tmp_score)
        
        #print ("**********************************************")
        #print ("                                              ")
        #print ("maximum Score for word ", word , " is " , score)        
    else:
        score = 0        
    return score

#-------------------BM25 Semantic Similarity Model  Begin-------------#
    
def GetBM25Score(Query, Passage,embeddingsType, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength    
    #print ("Query :" , Query)
    #print (' ')    
    #print ("Passage :",Passage)
    query_words= Query.strip().lower().split(delimiter)
    passage_words = Passage.strip().lower().split(delimiter)
    #passageLen = len(passage_words)
    
    #for word in set(query_words):
    docCosineScore = {}    
    
    # Lets assume query (Ss) length is small compared to passage (Sl) , otherwise just swap list , so that query always contains small text
    if len(query_words) > len(passage_words):
        query_words , passage_words = passage_words , query_words
    
    query_length = len(query_words)
    
    for word in set(passage_words):
        docCosineScore[word] = GetSemanticSimilarityScore(word, query_words,embeddingsType) # retuns the max score        
        
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docCosineScore[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docCosineScore[word]) + k1*(1 - b + b*query_length/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score
"normalisation  ---------  https://stats.stackexchange.com/questions/171589/normalised-score-for-bm25 "

#-------------------BM25 Semantic Similarity Model End-------------#


#Before calling BM25 Similarity, create IDF dictonary, which will be used for calculating BM25 Score
IDF_Generator(InputData)
#create a list to store cosine similarity Values:    
bm25_similarity_list = []

"""
bm25_sim_mat = pd.DataFrame(columns= [x for x in range(1,len(InputData)+1)])
temp_score = []
lno = 0
"""
def RunSimilarityOnCorpus(similarityMethod,embeddingsType):
    
    temp_score = []
    lno = 0
    df = pd.DataFrame(columns= [x for x in range(1,len(InputData)+1)])
    for i in range(len(InputData)):
        for j in range(len(InputData)):
            #print (i, " : ", j)
            if similarityMethod == 'BM25':
                score = GetBM25Score(InputData[i],InputData[j],embeddingsType)
            elif similarityMethod == 'AvgVec':
                score = GetAvgOfVectorsSimilarityScore(InputData[i],InputData[j],embeddingsType)
            elif similarityMethod == 'WeiSumVec':
                score = GetWeightedSumOfVectorsSimilarityScore(InputData[i],InputData[j],embeddingsType)
        
            temp_score.append(score)
            lno+=1
            if(lno%len(InputData) == 0):
                #print (lno)
                #print ("Inserting row into data frame")
                df.loc[i]=temp_score
               #print ("clearing temp score list")
                temp_score = []
               
    return df
        
"Make Sure that you've stored all the embeddings in dictionary form before calling below functions"

# -------------Similarity Matrix with Glove Embeddings----------------#

#glove_bm25_sim_mat = RunSimilarityOnCorpus('BM25',GloveEmbeddings)
#Glove_bm25_sim_mat = normalize(glove_bm25_sim_mat, axis=1, norm='l1')           #normalized BM25 matrix   

Glove_bm25_sim_mat = RunSimilarityOnCorpus('BM25',GloveEmbeddings)
Glove_avg_vec_sim_mat = RunSimilarityOnCorpus('AvgVec',GloveEmbeddings)
Glove_weighted_sum_vec_sim_mat = RunSimilarityOnCorpus('WeiSumVec',GloveEmbeddings)

# -------------Similarity Matrix with Glove Embeddings----------------#


#----------------Similarity Matrix with Word2Vec embeddings---------------#

#word2Vec_bm25_sim_mat = RunSimilarityOnCorpus('BM25',Word2vecEmbeddings)
#Word2Vec_bm25_sim_mat = normalize(word2Vec_bm25_sim_mat, axis=1, norm='l1')           #normalized BM25 matrix          

Word2Vec_bm25_sim_mat = RunSimilarityOnCorpus('BM25',Word2vecEmbeddings)
Word2Vec_avg_vec_sim_mat = RunSimilarityOnCorpus('AvgVec',Word2vecEmbeddings)
Word2Vec_weighted_sum_vec_sim_mat = RunSimilarityOnCorpus('WeiSumVec',Word2vecEmbeddings)

#----------------Similarity Matrix with Word2Vec embeddings---------------#


#-----Similarity Matrix with FastText embeddings----------------#

#fastText_bm25_sim_mat = RunSimilarityOnCorpus('BM25',FastTextEmbeddings)
#FastText_bm25_sim_mat = normalize(fastText_bm25_sim_mat, axis=1, norm='l1')     #normalized BM25 matrix

FastText_bm25_sim_mat = RunSimilarityOnCorpus('BM25',FastTextEmbeddings)
FastText_avg_vec_sim_mat = RunSimilarityOnCorpus('AvgVec',FastTextEmbeddings)
FastText_weighted_sum_vec_sim_mat = RunSimilarityOnCorpus('WeiSumVec',FastTextEmbeddings)

#-----Similarity Matrix with FastText embeddings----------------#
       
"""
#-------------Saving all the results-------------------#
     
print (Glove_bm25_sim_mat.shape)
print (Glove_avg_vec_sim_mat.shape)
print (Glove_weighted_sum_vec_sim_mat.shape)

Glove_bm25_sim_mat.to_csv("glove-BM25_score.csv")
Glove_avg_vec_sim_mat.to_csv("glove-Avg_Vec_score.csv")
Glove_weighted_sum_vec_sim_mat.to_csv("glove-weighted_sum_sim_score.csv") 

print (Word2Vec_bm25_sim_mat.shape)
print (Word2Vec_avg_vec_sim_mat.shape)
print (Word2Vec_weighted_sum_vec_sim_mat.shape)

Word2Vec_bm25_sim_mat.to_csv("Word2Vec-BM25_score.csv")
Word2Vec_avg_vec_sim_mat.to_csv("Word2Vec-Avg_Vec_score.csv")
Word2Vec_weighted_sum_vec_sim_mat.to_csv("Word2Vec-weighted_sum_sim_score.csv") 

print (FastText_bm25_sim_mat.shape)
print (FastText_avg_vec_sim_mat.shape)
print (FastText_weighted_sum_vec_sim_mat.shape)

FastText_bm25_sim_mat.to_csv("FastText-BM25_score.csv")
FastText_avg_vec_sim_mat.to_csv("FastText-Avg_Vec_score.csv")
FastText_weighted_sum_vec_sim_mat.to_csv("FastText-weighted_sum_sim_score.csv") 

#-------------Saving all the results-------------------#
"""

"""
X=np.concatenate([Glove_bm25_sim_mat,Word2Vec_bm25_sim_mat,FastText_bm25_sim_mat,
    Glove_avg_vec_sim_mat,Word2Vec_avg_vec_sim_mat,FastText_avg_vec_sim_mat,
    Glove_weighted_sum_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat,FastText_weighted_sum_vec_sim_mat], axis=0)#.reshape(-1,1)
X1_bm25withall          =    np.concatenate([Glove_bm25_sim_mat,Word2Vec_bm25_sim_mat,FastText_bm25_sim_mat], axis=0)#.reshape(-1,1)
X2_avgvec_withall       =    np.concatenate([Glove_avg_vec_sim_mat,Word2Vec_avg_vec_sim_mat,FastText_avg_vec_sim_mat], axis=0)#.reshape(-1,1)
X3_avgweighted_withall  =    np.concatenate([Glove_weighted_sum_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat,FastText_weighted_sum_vec_sim_mat], axis=0)#.reshape(-1,1)
X4_glovewithall         =    np.concatenate([Glove_bm25_sim_mat,Glove_avg_vec_sim_mat,Glove_weighted_sum_vec_sim_mat], axis=0)#.reshape(-1,1)
X5_Word2vecwithall      =    np.concatenate([Word2Vec_bm25_sim_mat,Word2Vec_avg_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat], axis=0)#.reshape(-1,1)
X6_fastextwithall       =    np.concatenate([FastText_bm25_sim_mat,FastText_avg_vec_sim_mat,FastText_weighted_sum_vec_sim_mat], axis=0)#.reshape(-1,1)
X7_glovebm25            =    np.array(Glove_bm25_sim_mat)
X8_glove_avgvec         =     np.array(Glove_avg_vec_sim_mat)
X9_glove_weighted       =     np.array(Glove_weighted_sum_vec_sim_mat)
X10_word2vec_bm25       =     np.array(Word2Vec_bm25_sim_mat)
X11_word2vec_avgvec     =     np.array(Word2Vec_avg_vec_sim_mat)
X12_word2vec_weighted   =     np.array(Word2Vec_weighted_sum_vec_sim_mat)
X13_fasttext_bm25       =     np.array(FastText_bm25_sim_mat)
X14_fasttext_avgvec     =     np.array(FastText_avg_vec_sim_mat)
X15_fasttext_weighted   =     np.array(FastText_weighted_sum_vec_sim_mat)

"""
"""
X=np.array([Glove_bm25_sim_mat],[Word2Vec_bm25_sim_mat],[FastText_bm25_sim_mat],
    [Glove_avg_vec_sim_mat],[Word2Vec_avg_vec_sim_mat],[FastText_avg_vec_sim_mat],
    [Glove_weighted_sum_vec_sim_mat],[Word2Vec_weighted_sum_vec_sim_mat],[FastText_weighted_sum_vec_sim_mat])
#X= Glove_bm25_sim_mat+Word2Vec_bm25_sim_mat+FastText_bm25_sim_mat+Glove_avg_vec_sim_mat+Word2Vec_avg_vec_sim_mat+FastText_avg_vec_sim_mat+Glove_weighted_sum_vec_sim_mat+Word2Vec_weighted_sum_vec_sim_mat+FastText_weighted_sum_vec_sim_mat
X1_bm25withall          =    np.array([Glove_bm25_sim_mat],[Word2Vec_bm25_sim_mat],[FastText_bm25_sim_mat])
X2_avgvec_withall       =   np.array([Glove_avg_vec_sim_mat],[Word2Vec_avg_vec_sim_mat],[FastText_avg_vec_sim_mat])
X3_avgweighted_withall  =   np.array([Glove_weighted_sum_vec_sim_mat],[Word2Vec_weighted_sum_vec_sim_mat],[FastText_weighted_sum_vec_sim_mat])
X4_glovewithall         =   np.array( [Glove_bm25_sim_mat],[Glove_avg_vec_sim_mat],[Glove_weighted_sum_vec_sim_mat])
X5_Word2vecwithall      =    np.array([Word2Vec_bm25_sim_mat],[Word2Vec_avg_vec_sim_mat],[Word2Vec_weighted_sum_vec_sim_mat])
X6_fastextwithall       =   np.array( [FastText_bm25_sim_mat],[FastText_avg_vec_sim_mat],[FastText_weighted_sum_vec_sim_mat])
X7_glovebm25            =    np.array([Glove_bm25_sim_mat])
X8_glove_avgvec         =    np.array([Glove_avg_vec_sim_mat])
X9_glove_weighted       =    np.array([Glove_weighted_sum_vec_sim_mat])
X10_word2vec_bm25       =    np.array([Word2Vec_bm25_sim_mat])
X11_word2vec_avgvec     =    np.array([Word2Vec_avg_vec_sim_mat])
X12_word2vec_weighted   =    np.array([Word2Vec_weighted_sum_vec_sim_mat])
X13_fasttext_bm25       =    np.array([FastText_bm25_sim_mat])
X14_fasttext_avgvec     =    np.array([FastText_avg_vec_sim_mat])
X15_fasttext_weighted   =    np.array([FastText_weighted_sum_vec_sim_mat])"""



X=[Glove_bm25_sim_mat,Word2Vec_bm25_sim_mat,FastText_bm25_sim_mat,
    Glove_avg_vec_sim_mat,Word2Vec_avg_vec_sim_mat,FastText_avg_vec_sim_mat,
    Glove_weighted_sum_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat,FastText_weighted_sum_vec_sim_mat]
#X= Glove_bm25_sim_mat+Word2Vec_bm25_sim_mat+FastText_bm25_sim_mat+Glove_avg_vec_sim_mat+Word2Vec_avg_vec_sim_mat+FastText_avg_vec_sim_mat+Glove_weighted_sum_vec_sim_mat+Word2Vec_weighted_sum_vec_sim_mat+FastText_weighted_sum_vec_sim_mat
X1_bm25withall          =    [Glove_bm25_sim_mat,Word2Vec_bm25_sim_mat,FastText_bm25_sim_mat]
X2_avgvec_withall       =    [Glove_avg_vec_sim_mat,Word2Vec_avg_vec_sim_mat,FastText_avg_vec_sim_mat]
X3_avgweighted_withall  =    [Glove_weighted_sum_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat,FastText_weighted_sum_vec_sim_mat]
X4_glovewithall         =    [Glove_bm25_sim_mat,Glove_avg_vec_sim_mat,Glove_weighted_sum_vec_sim_mat]
X5_Word2vecwithall      =    [Word2Vec_bm25_sim_mat,Word2Vec_avg_vec_sim_mat,Word2Vec_weighted_sum_vec_sim_mat]
X6_fastextwithall       =    [FastText_bm25_sim_mat,FastText_avg_vec_sim_mat,FastText_weighted_sum_vec_sim_mat]
X7_glovebm25            =    [Glove_bm25_sim_mat]
X8_glove_avgvec         =    [Glove_avg_vec_sim_mat]
X9_glove_weighted       =    [Glove_weighted_sum_vec_sim_mat]
X10_word2vec_bm25       =    [Word2Vec_bm25_sim_mat]
X11_word2vec_avgvec     =    [Word2Vec_avg_vec_sim_mat]
X12_word2vec_weighted   =    [Word2Vec_weighted_sum_vec_sim_mat]
X13_fasttext_bm25       =    [FastText_bm25_sim_mat]
X14_fasttext_avgvec     =    [FastText_avg_vec_sim_mat]
X15_fasttext_weighted   =    [FastText_weighted_sum_vec_sim_mat]



DataArray =[X, X1_bm25withall, X2_avgvec_withall,X3_avgweighted_withall,X4_glovewithall,X5_Word2vecwithall,
            X6_fastextwithall,X7_glovebm25,X8_glove_avgvec,X9_glove_weighted,X10_word2vec_bm25,X11_word2vec_avgvec
            ,X12_word2vec_weighted,X13_fasttext_bm25,X14_fasttext_avgvec,X15_fasttext_weighted]


#---------------shilloutee score---------#
def findCluster(X):
    for i in range(len(X)):
        X1 = np.array(X[i])
        #range_n_clusters = list (range(2,20))
        range_n_clusters = list(range(3,20))
        #print ("Number of clusters from 2 to 20: \n", range_n_clusters)
        max_score = 0
        no_of_clusters = 0
        
        for n_clusters in range_n_clusters:
            print(n_clusters)
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            preds = clusterer.fit_predict(X[i])
            print(np.unique(clusterer.labels_))
            #centers = clusterer.cluster_centers_
            score = silhouette_score (X1, clusterer.labels_)#, metric='euclidean')
            if score > max_score:
                max_score = score
                no_of_clusters = n_clusters
            print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))
        return no_of_clusters   
    #---------------shilloutee score---------#  
no_of_clusters = findCluster(X)
print(no_of_clusters)

#----------------------KMEANS clustering---------------------#

def kmeansClustering(X):
    for i in range(len(X)):
        X1 = np.array(X[i])
        #plt.scatter (X1[:,0],X1[:,1],label='true position')           #to visualize the data with original position
        kmeans = KMeans(n_clusters=no_of_clusters, random_state=5).fit(X1)
        #kmeans(flame, n_clusters=3)    
        #kmeans(agg, n_clusters=3)
        labels=kmeans.predict(X1)
        #print (kmeans.cluster_centers_)                   #centroid values the algorithm generated for final clusters
        #print (kmeans.labels_)                            #to see the labels for the data point
    
        #clustering = collections.defaultdict(list)
        #plt.scatter(X1[:,0],X1[:,1], c=kmeans.labels_, cmap='rainbow')     #to distinguish the colours
        #plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black') #to distnguish the centroids of clusters
        #plt.title("K-means")
        return labels
        #plt.show()

#----------------------KMEANS clustering---------------------#


#--------------DBSCAN Clustering--------------------------#    
def DBScanClustering(X):    
    for i in range(len(X)):
        X2 = np.array(X[i])
        db = DBSCAN(eps=1, min_samples=2, algorithm='auto', leaf_size=30, metric='euclidean',
                    metric_params=None, n_jobs=None, p=None).fit(X2)
        #print (db.labels_)
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #y_pred = db.fit_predict(X2)
        y_pred = db.labels_
        #plt.scatter(X2[:,0], X2[:,1],c=y_pred, cmap='rainbow')
        #plt.title("DBSCAN")
        return y_pred
        #plt.show()
    
#--------------DBSCAN Clustering--------------------------#   


#------------------Agglomarative clustering-----------------#
def AgglomarativeClustering(X):    
    for i in range(len(X)):
        X3 = np.array(X[i])
        hier = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None,linkage='ward', memory=None, n_clusters=no_of_clusters,
                        pooling_func='deprecated')
        z_pred = hier.fit_predict(X3)
        #print (hier.labels_)
        #plt.scatter(X3[:,0], X3[:,1],c=z_pred, cmap='rainbow')
        #plt.title("Hierarchical")
        return z_pred
        #plt.show()
 
#------------------Agglomarative clustering-----------------#
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
def SphericalClustering(X):    
    for i in range(len(X)):
        X4 = np.array(X[i])
        spher = make_pipeline(Normalizer(), KMeans(n_clusters=no_of_clusters))
        #spher = skm.SKM(k=10, assignment_change_eps=0.01, normalize=False,standardize=False, variance_explained=0.99, visualize=False)
        w_pred = spher.fit_predict(X4)
        print (spher.labels_)
        #plt.scatter(X3[:,0], X3[:,1],c=z_pred, cmap='rainbow')
        #plt.title("Hierarchical")
        return w_pred
        #plt.show()        
# Visualize cluster assignments (data must be tranformed using the SKM's learned PCA)
        #skm.display.visualize_clusters(s, s.pca.transform(data.T).T)
"""
#--------------Mean shift clustering-----------------#
        
"""
from sklearn.cluster import MeanShift
def Meanshift(X):
    for i in range(len(X)):
        X4 = np.array(X[i])
        mean = MeanShift(bandwidth=2, bin_seeding=False, cluster_all=True, min_bin_freq=1, seeds=None).fit(X4)
        w_pred= mean.labels_
        print (mean.labels_)
        return w_pred
"""

#--------------Mean shift clustering-----------------#



rand_accuracy_scores =[]
mutal_info_scores =[]
norm_mutual_info_scores = []
hom_v_measure_scores=[]
fowl_mall_scores=[]
sil_scores=[]
cal_scores=[]
max_score_array=[]
dav_scores =[]



#------------normal-------#

def predictModels1(DataArray):
    for i in range(0,len(DataArray)):
        kmeans_pred = kmeansClustering(DataArray[i])
        db_pred = DBScanClustering(DataArray[i])
        agg_pred = AgglomarativeClustering(DataArray[i])
        #spher_pred = SphericalClustering(DataArray[i])
        #mean_pred = Meanshift(DataArray[i])

        
        pred_array = [kmeans_pred, agg_pred, db_pred]
        for j in range(0,len(pred_array)):
            #rand_score = metrics.adjusted_rand_score(pred_array[j], labels_true)              #1
            mutual_info_score = metrics.adjusted_mutual_info_score(pred_array[j], labels_true)        #2     
            norm_mutual_info_score = metrics.normalized_mutual_info_score(pred_array[j], labels_true)            
            #hom_v_measure = metrics.homogeneity_completeness_v_measure(pred_array[j], labels_true)  #3            
            fowl_mall_score = metrics.fowlkes_mallows_score(pred_array[j], labels_true)              #4            
            #sil_score = metrics.silhouette_score(pred_array[j], labels_true, metric='euclidean')         #5            
            #cal_score = metrics.calinski_harabaz_score(pred_array[j], labels_true)
            #dav_score = metrics.davies_bouldin_score(pred_array[j], labels_true)
            #rand_accuracy_scores.append(rand_score)
            mutal_info_scores.append(mutual_info_score)
            norm_mutual_info_scores.append(norm_mutual_info_score)
            #hom_v_measure_scores.append(hom_v_measure)
            fowl_mall_scores.append(fowl_mall_score)
            #sil_scores.append(sil_score)
            #cal_scores.append(cal_score)
            #dav_scores.append(dav_score)
    max_score_array = [mutal_info_scores,norm_mutual_info_scores,fowl_mall_scores]
    return max_score_array 


max_array = predictModels1(DataArray)
max_score = 0
max_index = 0
for k in range(0,len(max_array)):
    if max(max_array[k]) > max_score:
        max_score = max(max_array[k])
        max_index = k
        
print("Score",max_score)
print("Index", max_index)    


#------------normal-------#

"""

#---------------precision re call-------#
 

named_array = ['X','X1_bm25withall','X2_avgvec_withall', 'X3_avgweighted_withall','X4_glovewithall','X5_Word2vecwithall',
            'X6_fastextwithall','X7_glovebm25','X8_glove_avgvec','X9_glove_weighted','X10_word2vec_bm25','X11_word2vec_avgvec'
            ,'X12_word2vec_weighted','X13_fasttext_bm25','X14_fasttext_avgvec','X15_fasttext_weighted']



pred_named_array = ['KMeans', 'DBScan', 'Agglomarative']

final_array = {}
def predictModels(DataArray):
    
    for i in range(0,len(DataArray)):
        kmeans_pred = kmeansClustering(DataArray[i])
        db_pred = DBScanClustering(DataArray[i])
        agg_pred = AgglomarativeClustering(DataArray[i])
        
        pred_array = [kmeans_pred, db_pred, agg_pred]
        
        for j in range(0,len(pred_array)):
             final_array[named_array[i]+' '+pred_named_array[j]] = precision_recall_fscore_support(pred_array[j],labels_true,average = 'weighted')
            
    return final_array    



max_array = predictModels(DataArray)
print(max_array)


#---------------precision re call-------#





#-------------table for precison recall-----------#
table = PrettyTable()

table.field_names =['DataSet Name', 'Precision','Recall','F1 Score']
def display_data(dict):
    for x,y in dict.items():
        table.add_row([x,y[0],y[1],y[2]])
    
    
display_data(max_array)   
print(table) 


def findMaxF1Score(dict):
    max_score = 0
    dataset = ''
    for x,y in dict.items():
        if y[2] > max_score:
            max_score = y[2]
            dataset = x
    return max_score, dataset        
    
print(findMaxF1Score(max_array))

max_score = 0
max_index = 0
for k in range(0,len(max_array)):
    if max(max_array[k]) > max_score:
        max_score = max(max_array[k])
        max_index = k
        
print("Score",max_score)
print("Index", max_index)

#-------------table for precison recall-----------#
"""