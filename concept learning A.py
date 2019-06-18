
# coding: utf-8

# In[1]:


f = open('9Cat-Train.labeled', 'r')
content = f.read()


# In[2]:


import numpy as np
contents = content.rstrip("\n")
line = contents.split('\n')

data = []
for n in line:
    data.append(n)

c = []
for a in data:
    b = a.split()
    c.append(b[1::2])


# In[3]:


# the size of the input space
first_q = 2**9
print first_q


# In[4]:


# the size of the concept space
second_q = 2**first_q
print len(str(abs(second_q)))


# In[5]:


# the size of the hypothesis space
print 1+3**(len(c[0])-1)


# In[6]:


# the size of the new hypothesis space
print 1+3**(len(c[0])-1+1)


# In[7]:


#the size of the newest hypothesis space
print 1+4*3**(len(c[0])-1-1)


# In[8]:


import random
import itertools
from itertools import chain

f = open("partA6.txt", "w+") 
# get the number of categories
num_categories = len(c[0])-1

    
# get the first survival case
for i in range(len(c)):
    if c[i][num_categories]=='Yes':
        hypothesis = (c[i])
        break
        


# Counter
k = 0

# a list to store results
result = []
results = []


for row in c:    
    # if survived, compare it with the hypothesis
    if row[num_categories] == 'Yes':
        x = zip(row, hypothesis)
        for i, pair in enumerate(x):
            if pair[0] != pair[1]:
                hypothesis[i] = '?'
            result.append(hypothesis)


    # append every 20, 40,... hypothesis
    k += 1
    if k % 20 == 0:
        print>>f,('\t'.join(result[k][:-1]))
#             print>>f,('\t'.join(result[k][:-1])) 
#         results.append(result[k])
# #          print>>f, result[k][:-1]
# for i in result:
#     print>>f,('\t'.join(i[:-1]))            


# print result


f.close()


# In[9]:


# get the final hypothesis
final_hypothesis = result[-1]

# seperate input and output of final hypothesis
hypothesis_val = final_hypothesis[:-1]


# In[11]:


# get the dev file
f1 = open('9Cat-Dev.labeled', 'r')
dev_content = f1.read()

# clean data
dev_contents = dev_content.rstrip("\n")
dev_line = dev_contents.split('\n')

dev_data = []
for n in dev_line:
    dev_data.append(n)

dev_c = []
for a in dev_data:
    b = a.split()
    row = b[1::2]
    dev_c.append(b[1::2])

# get the new list without output
dev_list = []
for row in dev_c:
    dev_list += [row[:-1]]
    
# Counter
correct = 0
total = 0

# get misclassification rate
p = 0
for row in dev_list:
    if p <= len(dev_c)-1:
        for n in [i for i, x in enumerate (hypothesis_val) if x == '?']:
            row[n] = '?'
        if row == hypothesis_val:
            if dev_c[p][-1] == 'Yes':
                correct += 1            
        else:
            if dev_c[p][-1] == 'No':
                correct += 1
    p += 1
    total += 1


misclassification_rate = (total - correct) / float(total)

print misclassification_rate


# In[716]:


import re
import sys

# get the input file
file_name = sys.argv[1]
fp = open (file_name)
input_content = fp.read()


input_contents = input_content.rstrip("\n")
input_line = input_contents.split('\n')

input_data = []
for n in input_line:
    input_data.append(n)

input_c = []
for a in input_data:
    b = a.split()
    row = b[1::2]
    input_c.append(b[1::2])

input_list = []
for row in input_c:
    input_list += [row[:-1]]
    

# print out the classification of each instance
p = 0
for row in input_list:
    if p <= len(input_c)-1:
        for n in [i for i, x in enumerate (hypothesis_val) if x == '?']:
            row[n] = '?'
        if row == hypothesis_val:
            print 'Yes'
        else:
            print 'No'
    p += 1

