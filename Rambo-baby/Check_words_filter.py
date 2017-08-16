import Util
import re
import csv
from Util import timing
import unicodedata


def searchWholeWord(w):
    w = unicode(w, "utf-8")
    w = unicodedata.normalize('NFD', w).encode('ascii', 'ignore')
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def read_words():
    lines = [line.rstrip('\n') for line in open('rambo_baby_words.txt')]
    return lines

@timing
def finding_matchs(sentence=str, words=[]):
    r = False
    w = None
    for word in words:
        return_value = searchWholeWord(word)(sentence)
        if return_value:
            r = True
            w = word
            break

    return r, w




def chek_stencence(sentence=str, words=[]):
    return any(word in sentence for word in words)

if __name__ == '__main__':
    words = read_words()
    words = [word.lower() for word in words]

    database, labels = Util.read_file('../Data/database.csv')

    false_positives = [database[i].lower() for i in range(len(database)) if not finding_matchs(database[i], words) and
                       labels[i][1] == 'Store']

    false_positives = list(set(false_positives))
    false_negatives = []
    matchs= []
    for i in range(len(database)):
        result, word = finding_matchs(database[i].lower(), words)

        if not result and labels[i][1] != "Store":
            false_negatives.append(database[i])
            matchs.append(word)

    #false_negatives = [database[i].lower() for i in range(len(database)) if finding_matchs(database[i], words) and
    #                   labels[i][1] != 'Store']

    false_negatives = list(set(false_negatives))

    with open('falsos positivos.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["Falsos Positivos", "Palavra encontrada"])

        for i in range(len(false_negatives)):
            spamwriter.writerow([false_negatives[i], matchs[i]])