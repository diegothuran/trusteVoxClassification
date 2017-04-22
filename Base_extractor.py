import sklearn
import os
import matplotlib.pyplot as plt
from Util import read_file, merge_lists, generate_n_gram_database
import cPickle

database = []
labels = []

for path_to_file in os.listdir("Data"):
    data, labe = read_file(os.path.join("Data",path_to_file))
    database.append(data)
    labels.append(labe)

database = merge_lists(database)
labels = merge_lists(labels)

for i in range(5):
    generate_n_gram_database(database, n=i+1, file_name='n_gram_databases/dabase_with_n_gram_'+ str(i+1) +'.data')

