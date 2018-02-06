# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 01:51:30 2018
Get META data associated with primary paper and citations based on citation tree
@author: yiqin
"""

def makeDOI(x):
    doi = str(re.findall(r"'(.*?)'", x, re.DOTALL))
    doi="https://doi.org/" + doi.replace("[","").replace("'","").replace("]","")
    return doi

def strip(x):
    string = str(re.findall(r"'(.*?)'",x, re.DOTALL)).replace("[","").replace("'","").replace("]","").replace("  ","").replace(" ","")
    return string

##Read in entire citation tree
import yaml 
with open("H:/Insight/bib_cites_parsed.yaml") as f:
    try:
        dataMap = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        
        ##Parse the citation tree into lists
import re

#Get the keys
key_list=list(dataMap.keys())

#Pull out APJ citations not in 2017 and 2018 (those are behind pay walls)
# In[29]:

citations=[]
dois=[]
primary=[] 

for key in key_list:
    for i in range(len(dataMap[key]['Citation'].split(","))):
            if "ApJ." in dataMap[key]['Citation'].split(",")[i] and any(x in dataMap[key]['Citation'].split(",")[i] for x in list(map(str, range(1990,2017)))):
                citations.append(dataMap[key]['Citation'].split(",")[i].replace("]","").replace("[",""))
                doi = str(re.findall(r"'(.*?)'", dataMap[key]['DOI'].split(",")[i], re.DOTALL))
                doi="https://doi.org/" + doi.replace("[","").replace("'","").replace("]","")
                dois.append(doi)
                primary.append(dataMap[key]['Primary'])


strip(primary[1])

# In[29]:

import ads

primary_list=list(set(primary))

primary_list = [x.split(",") for x in primary_list]

flat_list = [item for sublist in primary_list for item in sublist]

flat_bibs=[strip(x) for x in flat_list]

ads.config.token = YourToken

titles = []
acks = []
years = []
author = []
citecount = []
arxiv_classes = []
keywords = []
authors = []
bibgroups = []
dois = []
grants = []


for bibs in flat_bibs:
    #papers.append(ads.SearchQuery(bibcode=flat_bibs[i]))
    x = ads.SearchQuery(bibcode=bibs,fl=["title","ack","year","first_author","citation_count","arxiv_class","keyword","author","bibgroup","doi","grant","bibcode"])
    for paper in x:
        titles.append(paper.title)
        acks.append(paper.ack)
        years.append(paper.year)
        author.append(paper.first_author)
        citecount.append(paper.citation_count)
        arxiv_classes.append(paper.arxiv_class)
        keywords.append(paper.keyword)
        authors.append(paper.author)
        bibgroups.append(paper.bibgroup)
        dois.append(paper.doi)
        grants.append(paper.grant)
    print(bibs)
        
primary = {
    "primary_titles" : titles,
    "primary_acks" : acks ,
    "primary_years" : years,
    "primary_author" : author, 
    "primary_citecount" :citecount, 
    "primary_arxiv_classes" :arxiv_classes,
    "primary_keywords" :keywords,
    "primary_authors" :authors,
    "primary_bibgroups" :bibgroups,
    "primary_dois" :dois, 
    "primary_grants" :grants,
    "primary_bibs":flat_bibs
            }

primary = pd.DataFrame(primary)

primary.to_csv("H:/Insight/Primary.csv")        


# In[Citations]:

import ads

citations_list=list(set(citations))

citations_list = [x.split(",") for x in citations_list]

flat_list = [item for sublist in citations_list for item in sublist]

flat_bibs=[strip(x) for x in flat_list][1378:]

ads.config.token = YourToken

titles = []
acks = []
years = []
author = []
citecount = []
arxiv_classes = []
keywords = []
authors = []
bibgroups = []
dois = []
grants = []
f_bibs = []


for bibs in flat_bibs:
    #papers.append(ads.SearchQuery(bibcode=flat_bibs[i]))
    x = ads.SearchQuery(bibcode=bibs,fl=["title","ack","year","first_author","citation_count","arxiv_class","keyword","author","bibgroup","doi","grant","bibcode"])
    for paper in x:
        titles.append(paper.title)
        acks.append(paper.ack)
        years.append(paper.year)
        author.append(paper.first_author)
        citecount.append(paper.citation_count)
        arxiv_classes.append(paper.arxiv_class)
        keywords.append(paper.keyword)
        authors.append(paper.author)
        bibgroups.append(paper.bibgroup)
        dois.append(paper.doi)
        grants.append(paper.grant)
        f_bibs.append(paper.bibcode)
    print(bibs)
        
citations = {
    "citations_titles" : titles,
    "citations_acks" : acks ,
    "citations_years" : years,
    "citations_author" : author, 
    "citations_citecount" :citecount, 
    "citations_arxiv_classes" :arxiv_classes,
    "citations_keywords" :keywords,
    "citations_authors" :authors,
    "citations_bibgroups" :bibgroups,
    "citations_dois" :dois, 
    "citations_grants" :grants,
    "citations_bibs":f_bibs
            }


citations = pd.DataFrame(citations)

citations.to_csv("H:/Insight/citations.csv")   

