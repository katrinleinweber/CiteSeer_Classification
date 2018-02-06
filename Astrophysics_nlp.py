#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:18:18 2018
Creating text based features from full text
@author: yiqinshen
"""
# In[Definie function to delete brakets and quotes]:

import pandas as pd

def strip(x):
    string = str(re.findall(r"'(.*?)'",x, re.DOTALL)).replace("[","").replace("'","").replace("]","")
    return string

# In[Read in file with labels attached]:

text = pd.read_csv("/Users/yiqinshen/Dropbox/merge_by_primary.csv", encoding='latin-1')

text["primary_bibs"]=list(map(lambda x:x.replace("\r\r",""),text["primary_bibs"]))

text["citation_bibs"]=list(map(lambda x:x.replace(" ",""),text["citation_bibs"]))


for i in range(len(text["figures_text"])):
    if isinstance(text["figures_text"][i], str):
        text["citation_text"][i] = text["citation_text"][i] + text["figures_text"][i]


# In[Cleaning up the text file]:

################NLP Data_Preprocessing#################
###################################################

from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import gensim 

#Pull out all the author names, so they are not counted in tokenized words
    
first_author_stopword = []
for i in range(len(text['citation_text'])):
    first_author_stopword.append(text['First_Author_Cite'][i].replace(" et al","").lower())
first_author_stopword = list(set(first_author_stopword))


################Tokenize The Sentence Surrounding the citation#################

#Delete stop words, including standard English stop word dictionary, and list of authors
final_tokens = []
for i in range(len(text['citation_text'])):
    article_tokens = gensim.utils.simple_preprocess(text['citation_text'][i])
    cleaned_tokens = []
    stop_words = stopwords.words('english')+["eg","et","el","al","gil","paz"] + first_author_stopword
    for token in article_tokens:
        if token not in stop_words:
            cleaned_tokens.append(token)
            
##Didn't do lemmatization, as a lot of specialized astronomy words could not be lemmatized correctly            

    #lemmatizer = WordNetLemmatizer()
    #lemmatized_tokens = []        
    #for token in cleaned_tokens:
        #lemmatized_tokens.append(lemmatizer.lemmatize(token))
    
    final_tokens.append(cleaned_tokens)
    
    #print(final_tokens)
    
################Tokenize tile of each primary publication#################
 
title_tokens_primary=[]
for i in range(len(text['primary_titles'])):   
    article_tokens = gensim.utils.simple_preprocess(str(text['primary_titles'][i]))
    #Stop words
    cleaned_tokens = []
    stop_words = stopwords.words('english')+["eg","et","el","al"]
    for token in article_tokens:
        if token not in stop_words:
            cleaned_tokens.append(token)
    
    title_tokens_primary.append(cleaned_tokens)
    
 
################Tokenize tile of each citation#################   
title_tokens_citation=[]
for i in range(len(text['primary_titles'])):

    #article tokens    
    article_tokens = gensim.utils.simple_preprocess(str(text['citation_titles'][i]))
 
    #Stop words
    cleaned_tokens = []
    stop_words = stopwords.words('english')+["eg","et","el","al"] 
    for token in article_tokens:
        if token not in stop_words:
            cleaned_tokens.append(token)
    
    title_tokens_citation.append(cleaned_tokens)


################Tokenize key words#################   
keyword_tokens = []
for i in range(len(text['citation_keywords'])):
    #article tokens    
    article_tokens = gensim.utils.simple_preprocess(str(text['citation_keywords'][i]))
    #Stop words
    cleaned_tokens = []
    stop_words = stopwords.words('english')+["eg","et","el","al"] #+ first_author_stopword
    for token in article_tokens:
        if token not in stop_words:
            cleaned_tokens.append(token)
    keyword_tokens.append(list(set(cleaned_tokens)))



################Add those features to text#################   
text["final_tokens"]=final_tokens
text["title_tokens_primary"]=title_tokens_primary
text["title_tokens_citation"]=title_tokens_citation
text["keyword_tokens"]=keyword_tokens

# In[Feature Construction]:


from numpy import nan
import numpy as np
import math
import Levenshtein

# Calculating the Levenshtein similarity between the primary and citation titles

title_distance = []
for i in range(len(text["citation_titles"])):
    temp = Levenshtein.distance(str(text["citation_titles"][i]), str(text["primary_titles"][i])) 
    title_distance.append(temp)
    
text["title_distance"]=title_distance   

# Feature Construction: The number associated with the section the citation first appeared in

text["citation_section"]=text["citation_section"].fillna(value=0)


##Mean Imputation of missing ection number
section_num = []
for i in range(len(text['citation_section'])):
    if text['citation_section'][i]!= 0:   
        section_num.append(int(text['citation_section'][i].split(".")[0]))
    else:
        section_num.append(2)

# Whether the citation appeared in figure text
figures_text = []
for i in range(len(text['figures_text'])):
    if isinstance(text['figures_text'][i], str):   
        figures_text.append("1")
    else:
        figures_text.append("0")


# Create whether HLSP was ever mentioned in the full text
whether_hlsp = []
for i in range(len(text['hlsp_text'])):
    if isinstance(text['hlsp_text'][i], str):   
        whether_hlsp.append("1")
    else:
        whether_hlsp.append("0")

# Adding three variables to dataframe
text['whether_hlsp']=whether_hlsp
text['figures_text']=figures_text
text['section_num']=section_num

##Compute whether author in the citation author list is also in the primary author list

def last_name(x):
    out = [y for y in strip(x).split(", ") if "." not in y]
    return out

author_overlap=[]
for i in range(len(text['citation_author'])):
    if isinstance(text['citation_author'][i], str) and isinstance(text['primary_author'][i], str):
        S1=set(last_name(text['citation_author'][i]))
        S2=set(last_name(text['primary_author'][i]))
        author_overlap.append(S1.intersection(S2))
    else:
        author_overlap.append({})

author_overlap_new=[]
for i in range(len(text['citation_author'])):
    if author_overlap[i] == set():
        author_overlap_new.append(0)
    else:
        author_overlap_new.append(1)

#Compute whether grants have HLSP related grants in it
        
grants=[]
for i in range(len(text['citations_grants'])):
    if isinstance(text['citations_grants'][i], str):
        if any(x in text['citations_grants'][i] for x in ['NAS5-26555','NNX09AF08G']):
            grants.append(1)
        else:
            grants.append(0)
    else:
        grants.append(0)

#Compute acknowledgement section has HLSP mentioning 
        
acks=[]
for i in range(len(text['citations_acks'])):
    if isinstance(text['citations_acks'][i], str):
        if any(x in text['citations_acks'][i] for x in ['HLSP','high level science product','High Level Science Product','MAST','Mikulski Archive for Space Telescopes','NAS5-26555','NNX09AF08G']):
            acks.append(1)
        else:
            acks.append(0)
    else:
        acks.append(0)
        
        

text['whether_overlap']=author_overlap_new
text['whether_grants']=grants
text['whether_acks']=whether_hlsp
