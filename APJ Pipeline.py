# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:57:47 2018
Read in API data and get full text
@author: yiqin
"""


import urllib
import csv
from bs4 import BeautifulSoup
import re
import requests
import pandas as pd

from os import listdir
from os.path import isfile, join


merge_by_primary = pd.read_csv("XXXX",encoding="latin1")

citation_list = pd.read_csv("XXXXXX",encoding="latin1")



citation_sections = []
figures_texts = []
citation_texts = []
hlsp_texts = []
locations = []
citation_bibs=[]
primary_bibs=[]
outcomes=[]
authors=[]


for i in range(len(citation_list)):
    try:
        #Get soup object of the full page
        r = requests.get(merge_by_primary["dois"][i])
        soup= BeautifulSoup(r.text,'html.parser')
        
        if int(merge_by_primary["year_citation"][i]) >= 2009:
            
            citation_section=[]
            for j in range(len(soup.find_all('h2',class_='header-anchor'))):
                try:
                    sectionhead = soup.find_all('h2',class_='header-anchor')[j].string
                    start = soup.find("h2", text= sectionhead)
                    div = start.find_next_sibling("div").text.replace("\xa0"," ")     
                    if (merge_by_primary["First_Author_Cite"][i] in div):
                        citation_section.append(sectionhead)
                except:
                    print('soup.find_all ERROR:',sectionhead)
                    continue

            figure_text=[]            
            ##Whether there are citations in the figurehead
            for k in range(len(soup.find_all('div',class_='article-text figure-caption'))):
                figurehead = soup.find_all('div',class_='article-text figure-caption')[k].text
                if (merge_by_primary["First_Author_Cite"][i] in figurehead):
                    figure_text.append(figurehead)

            ##replace the . in et al
            text = soup.get_text()
            text = text.replace("e.g.","eg").replace("i.e.","ie").replace("et al.","et al").replace("\xa0"," ")
            text = re.sub(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', '', text)

            # get sentence related to two cluster of words
            sentences = text.split('.')
            location = text.find(merge_by_primary["First_Author_Cite"][i])/len(text)
            citation_text=[]
            hlsp_text = []
            for st in sentences:
                if merge_by_primary["First_Author_Cite"][i] in st:
                    citation_text.append(st)
                if any(x in st for x in ['HLSP','high level science product','High Level Science Product','MAST','Mikulski Archive for Space Telescopes','NAS5-26555','NNX09AF08G']):
                    hlsp_text.append(st)
            
        else: 
            ##If the article is published in an earlier year, need to click on a fulltext button first
      
            button=soup.find("div", class_= "btn-multi-block mb-1")
          
            address = button.findAll("a")[1]['href']
    
            url_with_frame  = "http://iopscience.iop.org" + address
    
            r = requests.get(url_with_frame)
            soup1= BeautifulSoup(r.text,'html.parser')
    
            actual_url = url_with_frame + soup1.find_all('frame')[1].attrs['src']
    
            r = requests.get(actual_url)
            soup2 = BeautifulSoup(r.text,'html.parser')
          
            if (soup2.find("div") is not None):
                citation_section=[]
                for section in range(len(soup2.find_all('div',class_='sec'))):
                    if merge_by_primary["First_Author_Cite"][i] in soup2.find_all('div',class_='sec')[section].text:
                        citation_section.append(soup2.find_all('div',class_='sec')[section].text.split("\n")[0])
    
                figure_text=[]
                for section in range(len((soup2.find_all('div',class_='figlabel')))):        
                    figure_label = soup2.find_all('div',class_='figlabel')[section]
                    fig_text = figure_label.find_next_sibling(class_="p").text
                    if merge_by_primary["First_Author_Cite"][i] in fig_text:
                        figure_text.append(fig_text)
    
              ##Replace period in paranthesis with ** so it doesn't mess with our data
                text = soup2.get_text()
                text = text.replace("e.g.","eg")
                text = text.replace("i.e.","ie")
                text = text.replace("et al.","et al").replace("\xa0"," ")
                text = re.sub(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', '', text)
    
              # get sentence related to two cluster of words
                sentences = text.split('.')
                location = text.find(merge_by_primary["First_Author_Cite"][i])/len(text)
                citation_text=[]
                hlsp_text = []
                for st in sentences:
                    if merge_by_primary["First_Author_Cite"][i] in st:
                        citation_text.append(st.replace('\xa0',''))
                    if any(x in st for x in ['HLSP','high level science product','High Level Science Product','MAST','Mikulski Archive for Space Telescopes','NAS5-26555','NNX09AF08G']):
                        hlsp_text.append(st)
                      
                print(i)
                      
            if (soup2.find("div") is None):
                  
                  citation_section=[]
                
                  figure_text=[]
                  for section in range(len((soup2.find_all('table')))):             
                          fig_text = soup2.find_all('table')[section].text
                          if merge_by_primary["First_Author_Cite"][i] in fig_text and "Fig" in fig_text:
                                  figure_text.append(fig_text)
                
                  text = soup2.get_text()
                  text = text.replace("e.g.","eg")
                  text = text.replace("i.e.","ie")
                  text = text.replace("et al.","et al").replace("\xa0"," ")
                  text = re.sub(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', '', text)
                             
                  # get sentence related to two cluster of words
                  sentences = text.split('.')
                  location = text.find(merge_by_primary["First_Author_Cite"][i])/len(text)
                  citation_text=[]
                  hlsp_text = []
                  for st in sentences:
                          if merge_by_primary["First_Author_Cite"][i] in st:
                                  citation_text.append(st.replace('\xa0',''))
                          if any(x in st for x in ['HLSP','high level science product','High Level Science Product','MAST','Mikulski Archive for Space Telescopes','NAS5-26555','NNX09AF08G']):
                                  hlsp_text.append(st)
              
        citation_bibs.append(merge_by_primary["citation"][i])
        primary_bibs.append(merge_by_primary["primary"][i])  
        authors.append (merge_by_primary["First_Author_Cite"][i])
        figures_texts.append(' '.join(figure_text))
        citation_texts.append(' '.join(citation_text))
        hlsp_texts.append(' '.join(hlsp_text))
        citation_sections.append(' '.join(citation_section))
        locations.append(location)
        
    
        print(i)
  
    
    except:
        print('Error Rows:',i)
        continue
        
    
          to_predict = {
            "citation_bibs": citation_bibs,
            "figures_text": figures_texts,
            "citation_section":citation_sections,
            "citation_text":citation_texts, 
            "hlsp_text":hlsp_texts, 
            "location":locations,
            "primary_bibs":primary_bibs,
            "outcome":outcomes,
            "author":authors
        }
        
        to_predict = pd.DataFrame(to_predict)

       