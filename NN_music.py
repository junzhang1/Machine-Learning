#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import sys
import math
import numpy as np
import time
import random
import time

music_train_file = sys.argv[1]
with open(music_train_file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    music_train = []
    for i in readCSV:
        music_train.append(i)

# Get the original value
i = 0
value = []
year = []
length = []
jazz = []
rock = []
new_value = []
for row in music_train:
    i +=1
    if i > 1:
        year.append(row[0])
        length.append(row[1])
        jazz.append(row[2])
        rock.append(row[3])
        value.append(row)
min_year = min(year)
max_year = max(year)
min_length = min(length)
max_length = max(length)

# Do feature scaling and change "yes" to "1.0", "no" to "0.0"
year_new = []
length_new = []
jazz_new = []
rock_new = []
for i in year:
    year_new.append((float(i)-float(min_year))/(float(max_year)-float(min_year)))
for i in length:
    length_new.append((float(i)-float(min_length))/(float(max_length)-float(min_length)))
for i in jazz:
    if i == "yes":
        jazz_new.append(1.0)
    if i == "no":
        jazz_new.append(0.0)
for i in rock:
    if i == "yes":
        rock_new.append(1.0)
    if i == "no":
        rock_new.append(0.0)  
        
# Get the new training set
training = []
for i in range(len(year_new)):
    training.append(np.asarray([1.0, year_new[i], length_new[i], jazz_new[i], rock_new[i]]).reshape(1,-1))
training_data = np.asarray(training)

# Get keys
music_train_key_file = sys.argv[2]
with open(music_train_key_file) as csvfile:
    readCSV = csv.reader(csvfile)
    music_key = []
    for i in readCSV:
        music_key.append(i)
# Change "yes" to "1.0", "no" to "0.0"
key_new = []
for i in music_key:
    if i == ['yes']:
        key_new.append([1.0])
    if i == ['no']:
        key_new.append([0.0])
training_key = np.asarray(key_new,float)

# Get weights
music_weight_1 = sys.argv[4]
with open(music_weight_1) as csvfile:
    readCSV = csv.reader(csvfile)
    input_weight1 = []
    for i in readCSV:
        input_weight1.append(i)
input_weight = np.asarray(input_weight1,float)
# Get weights
music_weight_2 = sys.argv[5]
with open(music_weight_2) as csvfile:
    readCSV = csv.reader(csvfile)
    neuron_weight1 = []
    for i in readCSV:
        neuron_weight1.append(i)
neuron_weight = np.asarray(neuron_weight1,float)

training1 = []
for i in range(len(year_new)):
    training1.append([1.0, year_new[i], length_new[i], jazz_new[i], rock_new[i]])
training_data1 = np.asarray(training1)

last_loss = 13
for i in range(0,3054):
    # Get the hidden layer output    should be 100 * 3
    hidden_layer_output = 1/(1+(np.exp(-np.dot(training_data1, input_weight))))  
    # Get hidden layer output + x0
    new_hidden_layer_output = np.insert(hidden_layer_output, 0, 1.0, axis=1)
    # Get the output
    output = 1/(1+(np.exp(-np.dot(new_hidden_layer_output, neuron_weight))))  
    # Compute loss function
    k = output - training_key
    j = k * k
    sum_j = np.sum(j)
    loss_function_1 = 1/(2) * sum_j
    if loss_function_1 < last_loss:
        print(loss_function_1)
    else:
        break
    
    # Get delta output
    delta_output = -(-k)*output*(1-output)
    delta_hidden_layer =  new_hidden_layer_output * (1-new_hidden_layer_output) * delta_output * neuron_weight.T
    q = np.delete(delta_hidden_layer, 0, axis=1)
    input_weight_update = input_weight - 0.1 * np.dot(training_data1.T, q)
    hidden_layer_weight_update = neuron_weight - 0.1 * np.dot(new_hidden_layer_output.T, delta_output)
    last_loss = loss_function_1
    input_weight = input_weight_update
    neuron_weight = hidden_layer_weight_update
get_input_weight = input_weight
get_neuron_weight = neuron_weight
print('GRADIENT DESCENT TRAINING COMPLETED!')


# In[ ]:


# Get weights
music_weight_1 = sys.argv[4]
with open(music_weight_1) as csvfile:
    readCSV = csv.reader(csvfile)
    input_weight1 = []
    for i in readCSV:
        input_weight1.append(i)
input_weight = np.asarray(input_weight1,float)
# Get weights
music_weight_2 = sys.argv[5]
with open(music_weight_2) as csvfile:
    readCSV = csv.reader(csvfile)
    neuron_weight1 = []
    for i in readCSV:
        neuron_weight1.append(i)
neuron_weight = np.asarray(neuron_weight1,float)

s_sum_loss = 0
for w in range(0,15):
    for i in range(len(training_data)):
        s_hidden_layer_output = 1/(1+(np.exp(-np.dot(training_data[i], input_weight))))
        s_new_hidden_layer_output = np.insert(s_hidden_layer_output,0,1.0).reshape(1,-1)
        s_output = 1/(1+(np.exp(-np.dot(s_new_hidden_layer_output, neuron_weight))))  
        s_k = s_output - training_key[i]
        s_j = s_k * s_k
        s_loss_function_1 = 1/(2) * s_j
#         import pdb;pdb.set_trace()
        s_sum_loss += s_loss_function_1[0][0]
        s_delta_output = -(-s_k)*s_output*(1-s_output)
        s_new_hidden_layer_output1 = s_new_hidden_layer_output.reshape(1,-1)
        s_new_hidden_layer_output1 = s_new_hidden_layer_output1.T
        s_delta_hidden_layer =  s_new_hidden_layer_output * (1-s_new_hidden_layer_output) * s_delta_output * neuron_weight.T
        s_q = np.delete(s_delta_hidden_layer, 0, axis=1)
        s_input_weight_update = input_weight - 0.4 * training_data[i].T * s_q
        s_hidden_layer_weight_update = neuron_weight - 0.4 * s_new_hidden_layer_output1* s_delta_output
        input_weight = s_input_weight_update
        neuron_weight = s_hidden_layer_weight_update
    print(s_sum_loss)
    s_sum_loss = 0
print('STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.')


# In[ ]:


# Predicting
music_dev = sys.argv[3]
with open(music_dev) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    music_dev = []
    for i in readCSV:
        music_dev.append(i)
# Get the original value
i_d = 0
value_d = []
year_d = []
length_d = []
jazz_d = []
rock_d = []
new_value_d = []
for row in music_dev:
    i_d +=1
    if i_d > 1:
        year_d.append(row[0])
        length_d.append(row[1])
        jazz_d.append(row[2])
        rock_d.append(row[3])
        value_d.append(row)
min_year_d = min(year_d)
max_year_d = max(year_d)
min_length_d = min(length_d)
max_length_d = max(length_d)

# DO feature scaling and change "yes" to "1.0", "no" to "0.0"
year_new_d = []
length_new_d = []
jazz_new_d = []
rock_new_d = []
for i in year_d:
    year_new_d.append((float(i)-float(min_year_d))/(float(max_year_d)-float(min_year_d)))
for i in length_d:
    length_new_d.append((float(i)-float(min_length_d))/(float(max_length_d)-float(min_length_d)))
for i in jazz_d:
    if i == "yes":
        jazz_new_d.append(1.0)
    if i == "no":
        jazz_new_d.append(0.0)
for i in rock_d:
    if i == "yes":
        rock_new_d.append(1.0)
    if i == "no":
        rock_new_d.append(0.0)   
        
# Get the new training set
training_d = []
for i in range(len(year_new_d)):
    training_d.append([1.0, year_new_d[i], length_new_d[i], jazz_new_d[i], rock_new_d[i]])
training_data_d = np.asarray(training_d)
hidden_layer_output_d = 1/(1+(np.exp(-np.dot(training_data_d, get_input_weight))))
new_hidden_layer_output_d = np.insert(hidden_layer_output_d,0,1.0, axis=1)
output_d = 1/(1+(np.exp(-np.dot(new_hidden_layer_output_d, get_neuron_weight))))
for i in output_d:
    if i >= 0.5:
        print('yes')
    if i < 0.5:
        print('no')

