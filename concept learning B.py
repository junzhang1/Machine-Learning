
# coding: utf-8

# In[2]:


# the size of input space
input_space = 2*2*2*2
print input_space


# In[3]:


# the size of the concept space
concept_space = 2 ** 16
print concept_space


# In[4]:


# open the file
f = open('4Cat-Train.labeled','r')
content = f.read()


# In[5]:


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



# get the list of column value
d = map(list, zip(*c))



# In[6]:


# I got this library from here: https://docs.python.org/2/library/itertools.html
import itertools

# get unique value for each category
age_value = list(set(d[0]))
class_value = list(set(d[1]))
embarked_value = list(set(d[2]))
sex_value = list(set(d[3]))

# get the list of value of each input space
value_of_input_space = list(itertools.product(age_value, class_value, embarked_value, sex_value))


# turn value_of_input_space into list
value_of_input_space_list = []
for k in value_of_input_space:
    value_of_input_space_list.append(list(k))


# In[7]:


# get the value of each concept space
vs = list(itertools.product(["Yes","No"], repeat=input_space))

# turn vs into list
vs_list = []
for k in vs:
    vs_list.append(list(k))



# In[8]:


# I get the following package from here: https://docs.python.org/2/library/copy.html
import copy
vs_copy = copy.deepcopy(vs_list)


# In[9]:


# list－then－eliminate algorithm
input_c = []
output_c = []
for row in c:
    input_c = row[:len(row)-1]
    output_c = row[len(row)-1]
    for n in range(len(value_of_input_space_list)):
        if input_c == value_of_input_space_list[n]:
            for k in vs_list:
                if output_c == 'No' and k[n] == 'Yes':
                    vs_copy.remove(k)
                if output_c == 'Yes' and k[n] == 'No':
                    vs_copy.remove(k)
    
    vs_list = copy.deepcopy(vs_copy)
print (str(len(vs_list)))
                
            


# In[10]:


import sys

file_name = sys.argv[1]
fp = open(file_name)
content_test = fp.read()

contents_test = content_test.rstrip("\n")
line_test = contents.split('\n')

data_test = []
for n in line_test:
    data_test.append(n)

c_test = []
for a in data_test:
    b_test = a.split()
    c_test.append(b_test[1::2])



# get the list of column value
d_test = map(list, zip(*c_test))

# test the test file
for row in c_test:
    num_yes = 0
    input_c_test = row[:len(row)-1]
    for n in range(len(value_of_input_space_list)):
        if input_c_test == value_of_input_space_list[n]:
            for k in vs_list:
                if k[n] == 'Yes':
                    num_yes += 1
    num_no = len(vs_list) - num_yes
    print (str(num_yes) + ' ' + str(num_no))

