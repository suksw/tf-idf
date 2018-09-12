from nltk import FreqDist, OrderedDict
import nltk
import operator
from nltk.tokenize import RegexpTokenizer
import math

docs = ['document1', 'document2', 'document3']

infile = '/home/subhashinie/Documents/CSE_7/Data_Mining/IR_temp_project/document/'

def countinfile(filename):
    path = nltk.data.find(filename)
    raw = open(path, 'rU').read()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw)
    text = nltk.Text(tokens)
    words = [w.lower() for w in text]
    freqDist = FreqDist(words)
    return OrderedDict(sorted(freqDist.items(), reverse=True, key=operator.itemgetter(0))), len(words)

def voc_size(orderedDict, doc):
    print(doc+'-vocSize'+':'+str(len(orderedDict.keys())))


def get_tf(orderedDict, docTotal):
    tfDict = {}
    for word, count in orderedDict.items():
        #tfDict[word] = 1+math.log10(float(count)/docTotal)
        tfDict[word] = 1+math.log10(float(count))
    return tfDict

def n_containing(word, df_dict_array):
    return sum(1 for one_doc in df_dict_array if word in one_doc)

def get_IDF(df_dict_array, doc_index): #for one doc
    one_tf_dict = df_dict_array[doc_index]
    idfDict = dict.fromkeys(one_tf_dict.keys(), 1)
    for df_dict in df_dict_array: # for all 3 docs
        for word, count in df_dict.items():
            if count > 0 and (word in idfDict):
                idfDict[word] += 1  # every time a word is present in a doc
    for word, val in idfDict.items():
            idfDict[word] = math.log10(4/float(val))
    return idfDict

def get_top_ten(tf, idf):
    weights = {}
    for word, val in tf.items():
        weights[word] = val*idf[word]
    return sorted(weights.items(), reverse=True, key=operator.itemgetter(1))[:10]

def finish_part_1():
    tf_dict_array = []
    big_total = 0
    for doc in docs:
        vocDict, words_in_doc = countinfile(infile + doc + '.txt')
        big_total += words_in_doc
        voc_size(vocDict, doc) #prints voc size

        tf_dict = get_tf(vocDict, words_in_doc)
        list = sorted(tf_dict.items())
        print(doc+'-tf-'+list[72][0]+', '+str(tf_dict[list[72][0]]))
        print('/n')
        tf_dict_array.append(tf_dict)
    return tf_dict_array

#voc size and tf
all_tf_dicts = finish_part_1()

#idf and top ten
for i in range(3):
    idf_dict = get_IDF(all_tf_dicts, i)
    idf_list = sorted(idf_dict.items())
    print(docs[i] + '-idf: ' + idf_list[72][0] + ', ' + str(idf_dict[idf_list[72][0]]))

    top_ten_dict = get_top_ten(all_tf_dicts[i], idf_dict)
    top_ten = ''
    for top_word in top_ten_dict:
        top_ten += top_word[0] + ','
    print(docs[i] + '-tf-' + top_ten)


