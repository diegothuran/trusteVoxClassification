import sklearn
import os
import matplotlib.pyplot as plt
from Util import read_file, merge_lists

database = []
labels = []

for path_to_file in os.listdir("Data"):
    data, labe = read_file(os.path.join("Data",path_to_file))
    database.append(data)
    labels.append(labe)

database = merge_lists(database)
labels = merge_lists(labels)


print database