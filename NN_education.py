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

education_train_file = sys.argv[1]
with open(education_train_file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    education_train = []
    for i in readCSV:
        education_train.append(i)
        
# Get the original value
i = 0
value = []
M1 = []
M2 = []
P1 = []
P2 = []
F = []
new_value = []
for row in education_train:
    i +=1
    if i > 1:
        M1.append(row[0])
        M2.append(row[1])
        P1.append(row[2])
        P2.append(row[3])
        F.append(row[4])
        value.append(row)
        
# Do feature scaling
M1_new = []
M2_new = []
P1_new = []
P2_new = []
F_new = []
for i in M1:
    M1_new.append((float(i))/100)
for i in M2:
    M2_new.append((float(i))/100)
for i in P1:
    P1_new.append((float(i))/100)
for i in P2:
    P2_new.append((float(i))/100)
for i in F:
    F_new.append((float(i))/100)

# Get the new training set
training = []
for i in range(len(M1)):
    training.append(np.asarray([1.0, M1_new[i], M2_new[i], P1_new[i], P2_new[i], F_new[i]]).reshape(1,-1))
training_data = np.asarray(training)

# Get keys
education_train_key_file = sys.argv[2]
with open(education_train_key_file) as csvfile:
    readCSV = csv.reader(csvfile)
    education_key = []
    for i in readCSV:
        education_key.append([float(i[0])])
education_key1 = np.asarray(education_key,float)

min_key = min(education_key1)
max_key = max(education_key1)

# Do feature scaling
key_new = []
for i in education_key1:
    key_new.append([(float(i))/(100)])
    
training_key = np.asarray(key_new,float)

# Get weights
education_weight_1 = sys.argv[4]
with open(education_weight_1) as csvfile:
    readCSV = csv.reader(csvfile)
    input_weight1 = []
    for i in readCSV:
        input_weight1.append(i)
input_weight = np.asarray(input_weight1,float)
# Get weights
education_weight_2 = sys.argv[5]
with open(education_weight_2) as csvfile:
    readCSV = csv.reader(csvfile)
    neuron_weight1 = []
    for i in readCSV:
        neuron_weight1.append(i)
neuron_weight = np.asarray(neuron_weight1,float)

training1 = []
for i in range(len(M1_new)):
    training1.append([1.0, M1_new[i], M2_new[i], P1_new[i], P2_new[i], F_new[i]])
training_data1 = np.asarray(training1)

for i in range(0,4000):
    # Get the hidden layer output
    hidden_layer_output = 1/(1+(np.exp(-np.dot(training_data1, input_weight))))        
    # Get hidden layer output + x0
    new_hidden_layer_output = np.insert(hidden_layer_output,0,1.0, axis=1)
    # Get the output
    output = 1/(1+(np.exp(-np.dot(new_hidden_layer_output, neuron_weight))))        
    # Compute loss function
    k = output - training_key
    j = k * k
    sum_j = np.sum(j)
    loss_function_1 = 1/(2) * sum_j
    print(loss_function_1)
    # Get delta output
    delta_output = -(-k)*output*(1-output)
    # Get delta hidden layer
    delta_hidden_layer = new_hidden_layer_output * (1-new_hidden_layer_output) * delta_output * neuron_weight.T
    q = np.delete(delta_hidden_layer, 0, axis=1)
    # Get updated input weights
    input_weight_update = input_weight - 0.025412 * np.dot(training_data1.T, q)
    # Get new hidden layer weights
    hidden_layer_weight_update = neuron_weight - 0.025412 * np.dot(new_hidden_layer_output.T, delta_output)
    input_weight = input_weight_update
    neuron_weight = hidden_layer_weight_update
get_input_weight = input_weight
get_neuron_weight = neuron_weight
print('GRADIENT DESCENT TRAINING COMPLETED!')


# In[ ]:


# Get weights
education_weight_1 = sys.argv[4]
with open(education_weight_1) as csvfile:
    readCSV = csv.reader(csvfile)
    input_weight1 = []
    for i in readCSV:
        input_weight1.append(i)
input_weight = np.asarray(input_weight1,float)
# Get weights
education_weight_2 = sys.argv[5]
with open(education_weight_2) as csvfile:
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
        s_sum_loss += s_loss_function_1[0][0]
        s_delta_output = -(-s_k)*s_output*(1-s_output)
        s_new_hidden_layer_output1 = s_new_hidden_layer_output.reshape(1,-1)
        s_new_hidden_layer_output1 = s_new_hidden_layer_output1.T
        s_delta_hidden_layer = s_new_hidden_layer_output * (1-s_new_hidden_layer_output) * s_delta_output * neuron_weight.T
        s_q = np.delete(s_delta_hidden_layer, 0, axis=1)
        s_input_weight_update = input_weight - 0.4 * training_data[i].T * s_q
        s_hidden_layer_weight_update = neuron_weight - 0.4 * s_new_hidden_layer_output1 * s_delta_output
        input_weight = s_input_weight_update
        neuron_weight = s_hidden_layer_weight_update
    print(s_sum_loss)
    s_sum_loss = 0
print('STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.')


# In[ ]:


# Predicting
education_dev = sys.argv[3]
with open(education_dev) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    education_dev = []
    for i in readCSV:
        education_dev.append(i)

# Get the original value
i_d = 0
value_d = []
M1_d = []
M2_d = []
P1_d = []
P2_d = []
F_d = []
new_value_d = []
for row in education_dev:
    i_d +=1
    if i_d > 1:
        M1_d.append(row[0])
        M2_d.append(row[1])
        P1_d.append(row[2])
        P2_d.append(row[3])
        F_d.append(row[4])
        value_d.append(row)
        
# DO feature scaling and change "yes" to "1.0", "no" to "0.0"
M1_new_d = []
M2_new_d = []
P1_new_d = []
P2_new_d = []
F_new_d = []
for i in M1_d:
    M1_new_d.append((float(i))/100)
for i in M2_d:
    M2_new_d.append((float(i))/100)
for i in P1_d:
    P1_new_d.append((float(i))/100)
for i in P2_d:
    P2_new_d.append((float(i))/100)
for i in F_d:
    F_new_d.append((float(i))/100)
        
# # Get the new training set
training_d = []
for i in range(len(M1_new_d)):
    training_d.append([1.0, M1_new_d[i], M2_new_d[i], P1_new_d[i], P2_new_d[i], F_new_d[i]])
training_data_d = np.asarray(training_d)

hidden_layer_output_d = 1/(1+(np.exp(-np.dot(training_data_d, get_input_weight))))
new_hidden_layer_output_d = np.insert(hidden_layer_output_d,0,1.0, axis=1)
output_d = 100* 1/(1+(np.exp(-np.dot(new_hidden_layer_output_d, get_neuron_weight))))
for i in output_d:
    print(float(i))

