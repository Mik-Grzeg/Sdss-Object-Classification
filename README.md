# Sdss-Object-Classification

Start/Galaxy/Quasar classification

The goal of this project was to implement a neural network from scratch, that would classified a space object to be star/galaxy/quasar based on photometric and spectroscopic features.
The neural network is optimized with particle swarm optimization algorithm.


# Getting started
### Prerequisites
Run command below in order install packages that are imported in jupyter notebook.
```bash
pip install -r requirements.txt
```

# Data set 
### Collection
The data was acquired from SDSS skyserver (DR 12). Data Release 12 is the final data release of the SDSS-III, containing all SDSS observations through July 2014.
[Here](http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx) you can execute your sql queries.

### Description 
Data set used has around 20000 records, that are split into train, validation and test sets. 
The biggest impact on classification have photometric and spectroscopic variables.
Each record has 18 features, some of them are redundant and few of them are transformed. 

# Results
### Performance
Data set is divided into train, validation and test sets with proportions 64/16/20.
Roughly after 300 generations of pso, the classifier achieves accuracy of 96% on test set.

![info](./pictures/ngc60.jpg)
