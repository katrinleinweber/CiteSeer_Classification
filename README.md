
## CiteSeer Citation Classification

## Synopsis

As an Insight Data Science Fellow, I consulted for NASA's Space Telescope Sciences Institute to help them classify different types of citations of thier community-contributed data products (CCDPs). 

There are two major types of citations:

(a) the use of data in a CCDP collection (dataset based citation), e.g. "Using our star/galaxy separator algorithm on the image stacks created by Smith & Wesson (2016), we find that the number of galaxies relative to field stars increases dramatically with photometric depth."

(b) the mention of scientific results without analyzing the CCDP data (finding based citation), e.g. "As shown by Smith & Wesson (2016), the ratio of the number of galaxies relative to field stars increases dramatically with photometric depth." 

This repo obtains data and implement predictive model to distinguish between those two classes.

## Dependencies

Astrophysics API, information about usage avaialable at https://github.com/adsabs/adsabs-dev-api

## Components

Astrophysics_API.py: Interface with astrophysics API to obtain metadata (including DOI) associated with script.
APJ_Scrapping.py: Scrapes fulltext of the astrophysics journal based on DOI
Astrophysics_nlp: NLP word tokenizing and word feature construction
Astrophysics_embedding: From tokenzied words to word embeddings
Astrophysics_modeling: Random Forest Classification Model

## Contributors

Yiqin Alicia Shen






