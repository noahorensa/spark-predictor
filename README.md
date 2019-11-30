## Spark Predictor

Spark predictor is a project originally developed by Shivaram et al. Ernest, a framework
outlined by the original authors, predicts the running time of advanced analytics jobs
on distributed computing engines. This repo is built for predicting the running time
for Apache Spark.  

The framework builds performance models based on the behavior of the job on small
samples of data and then predicts its performance on larger datasets and cluster
sizes. To minimize the time and resources spent in building a model, Ernest
uses [optimal experiment design](https://en.wikipedia.org/wiki/Optimal_design),
a statistical technique that allows us to collect as few training points as
required.
 
For more details please see the original [paper](http://shivaram.org/publications/ernest-nsdi.pdf) and [talk slides](http://shivaram.org/talks/ernest-nsdi-2016.pdf) from NSDI 2016.
Original [repository](https://github.com/amplab/ernest).

### Installation

The requirements for this repo are in requirements.txt. The easiest way to install is: 
```
pip install -r requirements.txt
```

The code runs for Python 2.7

### Usage

1. Determining what sample data points to collect. To do this we will be using experiment design
   implemented in [expt_design.py](expt_design.py). This will return the set of training data points
   required to build a performance model.  
2. Collect running time for the set of training data points. This depends on the analytics 
   application that we are building model for.
3. Building a performance model and using it for prediction. To do this we create a CSV file with
   measurements from previous step and use [predictor.py](predictor.py). 
