import os
import csv

def read_file(path):
    database = []
    labels = []
    with open(path, "rb") as csv_file:
        next(csv_file)
        spamreader = csv.reader(csv_file, delimiter = ',')
        for row in spamreader:
            row = map(lambda x: x if x != '' else '0', row)
            database.append(row[0])
            labels.append(row[1:-1])

    return database, labels

def merge_lists(lists):
    return lists[0] + lists[1] + lists[2]