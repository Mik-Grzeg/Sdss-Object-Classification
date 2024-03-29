# Sdss-Object-Classification

<p align="center"> 
<img src="https://user-images.githubusercontent.com/58633804/116714174-c305d680-a9d5-11eb-8263-83827977d101.jpg">
</p>

Start/Galaxy/Quasar classification

A purpose of this project was to implement a neural network from scratch, that would classify a space object to be a star/galaxy/quasar based on photometric and spectroscopic features.

The neural network is optimized with particle swarm optimization algorithm. 

The network and optimization algorithm is implemented in **neuralnet.py**, 
Data analysis, feature engineering, training and testing the classifier is conducted in **Sdss_classifier.ipynb** notebook.

# Getting started
### Prerequisites
Run command below in order install packages that are imported in jupyter notebook.
```bash
pip install -r requirements.txt
```
Open the notebook and run all cells.


# Data set 
### Collection
The data was acquired from SDSS skyserver (DR 12). Data Release 12 is the final data release of the SDSS-III, containing all SDSS observations through July 2014.
[Here](http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx) you can execute your sql queries.

### Description 
Data set used in this project has around 20000 observations, that are split into train, validation and test sets. 
The biggest impact on classification have photometric and spectroscopic variables.
Each record has 18 features, some of them are redundant and few of them are transformed. 

# Results
### Performance
Data set is divided into train, validation and test sets with proportions 64/16/20.
Roughly after 300 generations of pso, the classifier achieves accuracy of 96% on test set.
