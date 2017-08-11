#coding: utf-8
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class PreProcesso (object):
    def __init__(self):
        self.acentos = ['á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ì', 'ò', 'ù', 'ã', 'ẽ', 'ĩ', 'õ', 'ũ', 'â', 'ê', 'î', 'ô','û','ç']
        self.s_acentos = ['a', 'e', 'i', 'o', 'u', 'a', 'e', 'i', 'o', 'u', 'a', 'e', 'i', 'o', 'u', 'a', 'e', 'i', 'o','u','c']
        self.stop_words = set(stopwords.words("portuguese"))
        self.more_stopwords = ['ja', 'q', 'd', 'ai', 'desse', 'dessa', 'disso', 'nesse', 'nessa', 'nisso', 'esse', 'essa', 'isso', 'so', 'mt', 'vc', 'voce', 'ne', 'ta', 'to', 'pq',
                     'cade', 'kd', 'la', 'e', 'eh', 'dai', 'pra', 'vai', 'olha', 'pois','fica', 'muito', 'muita', 'muitos', 'muitas', 'onde', 'mim', 'oi', 'ola', 'ate','com']
        self.stemmer = nltk.stem.RSLPStemmer()

    def tokenizarSetenca(self,texto):
        return word_tokenize(texto)

    def removerAcentos(self,texto):
        for i in range(0,len(self.acentos)):
            texto = texto.replace(self.acentos[i],self.s_acentos[i])
        return texto

    def removerStopWords(self,texto):
        texto = ' '.join([word for word in word_tokenize(texto) if word not in self.stop_words])
        texto = ' '.join([word for word in word_tokenize(texto) if word not in self.more_stopwords])
        return texto

    def removerSufixo(self,texto):
        texto_tratado=''
        for word in texto:
            texto_tratado= texto_tratado+' '+self.stemmer.stem(word)
        return texto_tratado

    def prePorcessar(self,texto):
        texto = texto.lower()
        texto = self.removerAcentos(texto)
        texto = self.removerStopWords(texto)
        texto = self.tokenizarSetenca(texto)
        texto = self.removerSufixo(texto)
        return texto