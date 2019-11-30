import csv
import sys

import numpy as np
from scipy.optimize import nnls

class Predictor(object):

  def __init__(self, training_data_in=[], data_file=None):
    ''' 
        Initiliaze the Predictor with some training data
        The training data should be a list of [mcs, input_fraction, time]
    '''
    self.training_data = []
    self.training_data.extend(training_data_in)
    if data_file:
      with open(data_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)    # skip header
        for row in reader:
          mc = int(row[0])
          scale = float(row[1])
          time = float(row[2])
          self.training_data.append([mc, scale, time])

  def add(self, mcs, input_fraction, time):
    self.training_data.append([mcs, input_fraction, time])

  def predict(self, input_fraction, mcs):
    ''' 
        Predict running time for given input fraction, number of machines.
    '''    
    test_features = np.array(self._get_features([input_fraction, mcs]))
    return test_features.dot(self.model[0])

  def predict_all(self, test_data):
    ''' 
        Predict running time for a batch of input sizes, machines.
        Input test_data should be a list where every element is (input_fraction, machines)
    '''    
    test_features = np.array([self._get_features([row[0], row[1]]) for row in test_data])
    return test_features.dot(self.model[0])

  def fit(self):
    print "Fitting a model with ", len(self.training_data), " points"
    labels = np.array([row[2] for row in self.training_data])
    data_points = np.array([self._get_features(row) for row in self.training_data])
    self.model = nnls(data_points, labels)
    # Calculate training error
    training_errors = []
    for p in self.training_data:
      predicted = self.predict(p[0], p[1])
      training_errors.append(predicted / p[2])

    print "Average training error %f%%" % ((np.mean(training_errors) - 1.0)*100.0 )
    return self.model[0]

  def num_examples(self):
    return len(self.training_data)

  def _get_features(self, training_point):
    mc = training_point[0]
    scale = training_point[1]
    return [1.0, float(scale) / float(mc), float(mc), np.log(mc)]

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print "Usage <predictor.py> <csv_train_file> <csv_predictions>"
    sys.exit(0)

  pred = Predictor(data_file=sys.argv[1])

  model = pred.fit()

  print "Model parameters:" + str(model)

  test_data = []
  scale = 0.125
  while scale <= 8:
    test_data += [[mc, scale] for mc in xrange(2, 129, 2)]
    scale *= 2

  predicted_times = pred.predict_all(test_data)


  predictions = []
  for i in xrange(0, len(test_data)):
    predictions.append({
      'Machines': test_data[i][0],
      'Scale': test_data[i][1],
      'Time': predicted_times[i]
    })

  print predictions[0]

  with open(sys.argv[2], 'w') as f:
    writer = csv.DictWriter(f, fieldnames=["Machines", "Scale", "Time"])
    writer.writeheader()
    writer.writerows(predictions)
