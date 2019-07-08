
# coding: utf-8

# In[21]:


# I got the import data code here: https://docs.python.org/2/library/csv.html
import csv
import numpy as np
import sys
import re
data = []
file_name = sys.argv[1]
with open(file_name) as csvfile:
     reader = csv.reader(csvfile)
     next(reader, None)
     for row in reader:
         data.append(row[-1])


# In[28]:


# Compute label entropy
def entropy(labels):
    num = len(labels)
    if num <= 1:
        return 0
    element,count = np.unique(labels,return_counts = True)
    prob = count / float(num)
    number = len(prob)
    if number <= 1:
        return 0
    entropy = np.sum([(-count[i]/ float(num))*np.log2(count[i]/float(num))for i in range(len(element))])

    return entropy
print (" ".join(["entropy:", str(entropy(data))]))


# In[ ]:


# Compute error rate   
def error_rate(labels):
    num = len(labels)
    element,count = np.unique(labels,return_counts = True)
    max_index = max(count)
    error = num - max_index
    error_rate = float(error) / num
    
    return error_rate
print (" ".join(["error:", str(error_rate(data))]))

