#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import sys


# In[2]:


trainfile = sys.argv[1]
# trainfile = "split.train"
# get file name/words
def getfile(files):
    file = []
    with open(files, encoding="latin-1") as f:
        for k in f:
            k = k.strip().lower()
            file.append(k)
    return file
train_file_name = getfile(trainfile)


# In[3]:


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


log_odd_ratio = []
for k in words:     # words must be in the training set
    try:
        lib = math.log((dict_lib[k]+1.0)/(float(lib_length) + float(words_length)))
    except:       # words not in liberal
        lib = math.log(1.0/(float(lib_length) + float(words_length)))
    try:
        con = math.log((dict_con[k]+1.0)/(float(con_length) + float(words_length)))

    except:       # words not in conservative
        con = math.log(1.0/(float(con_length)+ float(words_length)))
    log_odd_ratio.append([lib - con, k])
log_odd_ratio.sort(reverse=True)


# In[7]:


for i in range(0,20):
    print(log_odd_ratio[i][1], " %.04f" % log_odd_ratio[i][0])
print('')
for i in range(1,21):
    print(log_odd_ratio[-i][1], " %.04f" % abs(log_odd_ratio[-i][0]))

