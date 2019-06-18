#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import math


# In[2]:


trainfile = sys.argv[1]
# trainfile = "split.train"
testfile  = sys.argv[2]
# testfile = "split.test"


# In[3]:


# get file name/words
def getfile(files):
    file = []
    with open(files, encoding="latin-1") as f:
        for k in f:
            k = k.strip().lower()
            file.append(k)
    return file
train_file_name = getfile(trainfile)

# get words in train file
word1 = []
for k in train_file_name:
    i = getfile(k)
    word1.append(i)

word=[]
for i in word1:
    for k in i:
        word.append(k)

# get unique word
words = set(word)

# get number of unique word
words_length = len(words)


# In[4]:


# calculate prior
condoc = 0
libdoc = 0
lib_all_word1 = []
con_all_word1 = []
for line in train_file_name:
    k = getfile(line)
    if line.startswith("con"):
        con_all_word1.append(k)
        condoc += 1
    else:
        lib_all_word1.append(k)
        libdoc += 1
prior_con = math.log(condoc/float(condoc + libdoc))
prior_lib = math.log(libdoc/float(condoc + libdoc))

# get all the words for lib and con
con_all_word = []
for i in con_all_word1:
    for k in i:
        con_all_word.append(k)
lib_all_word = []
for i in lib_all_word1:
    for k in i:
        lib_all_word.append(k)

#get number of word in lib and con
con_length = len(con_all_word)
lib_length = len(lib_all_word)


# In[5]:


# Counter the number of words
dict_lib = {}
dict_con = {}
for i in con_all_word:
    if i in dict_con:
        dict_con[i] += 1
    else:
        dict_con[i] = 1
for i in lib_all_word:
    if i in dict_lib:
        dict_lib[i] += 1
    else:
        dict_lib[i] = 1


# In[6]:


def prediction(test):
    nb_con = 0.0
    nb_lib = 0.0
    test_data = getfile(test)
    for k in test_data:
        if k in words:     # words must be in the training set
            try:
                lib = math.log((dict_lib[k]+1.0)/(float(lib_length) + float(words_length)))
            except:       # words not in liberal
                lib = math.log(1.0/(float(lib_length) + float(words_length)))
            nb_lib += lib
            try:
                con = math.log((dict_con[k]+1.0)/(float(con_length) + float(words_length)))

            except:       # words not in conservative
                con = math.log(1.0/(float(con_length)+ float(words_length)))
            nb_con += con
    nb_lib += prior_lib
    nb_con += prior_con
    if nb_lib <= nb_con:
        return "C"
    else:
        return "L"
predict = []


# In[7]:


test_label = []
test_file_name = getfile(testfile)
for line in test_file_name:
    predict.append(prediction(line))
    if line.startswith("con"):
        test_label.append("C")
    else:
        test_label.append("L")
    
total_test = len(test_label)

for line in predict:
    print(line)

correct = 0
for i,j in zip(predict, test_label):
    if i == j:
        correct +=1
accuracy = float(correct)/total_test
print("Accuracy: %.04f" % accuracy)

