import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_data(filename):
  import csv
  with open(filename, 'r', encoding='utf8') as data_file:
    data_reader = csv.reader(data_file)
    return [row for row in data_reader]

def read_labels(filename):
  with open(filename, 'r', encoding='utf8') as label_file:
    return [int(line.strip()) for line in label_file]

def write_predictions(filename, predictions):
  with open(filename, 'w', encoding='utf8') as pred_file:
    pred_file.writelines(map(lambda x: str(x) + '\n', predictions))

def get_titles(data):
  return [row[2] for row in data]

def run(training_fn, label_fn, testing_fn, pred_fn):
  training_data = read_data(training_fn)
  training_labels = read_labels(label_fn)

  vectorizer = CountVectorizer()
  training_vectors = vectorizer.fit_transform(get_titles(training_data))

  classifier = MultinomialNB()
  classifier.fit(training_vectors, training_labels)

  testing_data = read_data(testing_fn)
  testing_vectors = vectorizer.transform(get_titles(testing_data))

  y = classifier.predict_proba(testing_vectors)

  write_predictions(pred_fn, y[:,1]) # output the probability value of predicting '1'

def main(args):
  training_fn = args[0]
  label_fn    = args[1]
  testing_fn  = args[2]
  pred_fn     = args[3]

  run(training_fn, label_fn, testing_fn, pred_fn)

if __name__ == '__main__':
  main(sys.argv[1:])
