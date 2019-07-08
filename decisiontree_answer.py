
# coding: utf-8

# In[ ]:


import csv
import math
import sys
import numpy as np

trainData = sys.argv[1]
testData = sys.argv[2]

class tree(object):
    def __init__(self, path_name, min_mutual_information = 0.1, max_depth = 2, label = " Party ", 
                 plus = "democrat", minus = "republican"):
        self.label = label
        self.plus = plus
        self.minus = minus
        self.min_mutual_information = min_mutual_information
        self.max_depth = max_depth
       
        training_example = self._openFile(path_name)
        self.training_example = training_example
        self.root = Node(max_depth, plus, minus, training_example, min_mutual_information)
    
    def _openFile(self, path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            raw_data = []
            for row in reader:
                i = {}
                i["Output"] = row.pop(self.label)
                i["Attributes"] = row
                raw_data.append(i)
        return raw_data
    
    # Get classification error
    def _classficationError(self, raw_data):
        correct = 0
        number = len(raw_data)
        for i in raw_data:
            if self.root.classification_for_no_children(i) == i["Output"]:
                correct += 1
        error_rate = (number - correct) / float(number)
        return error_rate
    
    # Print the decision tree
    # I learned how to use self.root here:https://stackoverflow.com/questions/42743032/confused-by-root-vs-self-root-and-the-use-of-init-also-confused-with-defini
    def tree(self):
        self.root.tree()
            
    # Get the error rate on the training data
    def training_error(self):
        return self._classficationError(self.training_example)
    
    # Get the error rate on the testing data
    def testing_error(self, test_path):
        raw_data = self._openFile(test_path)
        return self._classficationError(raw_data)
        
    def result(self,test_path):
        self.tree()
        print (" ".join(["error(train):", str(self.training_error())]))
        print(" ".join(["error(test):", str(self.testing_error(test_path))]))


# In[ ]:


class Node(object):
    def __init__(self, max_depth, plus, minus, raw_data, min_mutual_information,
                 depth = 0, attribute = None, value = None):
        self.depth = depth
        self.max_depth = max_depth
        self.plus = plus
        self.minus = minus
        self.raw_data = raw_data
        self.min_mutual_information = min_mutual_information
        self.attribute = attribute
        self.value = value

        (splitAttribute, children) = self._getChildren()
        self.splitAttribute = splitAttribute
        self.children = children

        
        # Get number of plus and minus
        num = len(raw_data)
        num_minus = 0
        for row in raw_data:
            if row["Output"] == self.minus:
                num_minus += 1
        num_plus = num - num_minus
        self.output_plus = num_plus
        self.output_minus = num_minus
        
        # Get the classification
        if num_minus <= num_plus:
            self.classification = self.plus
        else:
            self.classification = self.minus


    # Compute entropy
    def _computeEntropy(self, raw_data):
        num = len(raw_data)
        num_plus = 0
        for row in raw_data:
            if row["Output"] == self.plus:
                num_plus += 1
        num_minus = num - num_plus
        prob_plus = num_plus / float(num)
        prob_minus = num_minus / float(num)
        labels = []
        for x in raw_data:
            labels.append(x["Output"])
        element,count = np.unique(labels,return_counts = True)
        prob = count / float(num)
        number = len(prob)
        if number <= 1:
            return 0
        entropy = -prob_plus*math.log(prob_plus,2) - prob_minus*math.log(prob_minus,2)
        return entropy
    
    # Compute conditional entropy
    def _conditionalEntropy(self, raw_data, attri):
        conditionalEntropy = 0
        num = len(raw_data)
        split = self._splitData(raw_data, attri)
        for i in split:
            entropyOfI = self._computeEntropy(split[i])
            prob = len(split[i]) / float(num)
            conditionalEntropy += prob * entropyOfI
        return conditionalEntropy
    
    # Split data
    def _splitData(self, raw_data, attri):
        split = {}
        for i in raw_data:
            value = i["Attributes"][attri]
            if value not in split:
                split[value] = []
            split[value].append(i)
        return split
    
    # Compute mutual information
    def _mutualInformation(self, raw_data, attri):
        mutualInformation = self._computeEntropy(raw_data) - self._conditionalEntropy(raw_data, attri)
        return mutualInformation
    
    def _noChildren(self):
        return self.children == None
    
    # Get children based on maximizing mutual information
    def _getChildren(self):
        if self.depth == self.max_depth:
            return (None, None)
        max_mutual_information = 0
        max_attribute = None
        for attri in self.raw_data[0]["Attributes"]:
            mutualInformation = self._mutualInformation(self.raw_data, attri)
            if mutualInformation > max_mutual_information:
                max_attribute = attri
                max_mutual_information = mutualInformation
        splitAttribute = max_attribute
        children = {}
        if max_mutual_information < self.min_mutual_information:
            return (None,None)
        new_example = self._splitData(self.raw_data, splitAttribute)
        for i in new_example:
            raw_data = new_example[i]
            children[i] = Node(self.max_depth, self.plus, self.minus, raw_data, self.min_mutual_information, 
                               depth = self.depth + 1, attribute = splitAttribute, value = i)
        return (splitAttribute, children)
    
    def _noParents(self):
        return (self.attribute == None)
    
    # Get classification for no-children node
    def classification_for_no_children(self, test):
        if self._noChildren():
            return self.classification
        i = test["Attributes"][self.splitAttribute]
        k = self.children[i]
        return k.classification_for_no_children(test)
    
    # Print the decision tree
    def tree(self):
        if self._noParents():
            print ("".join(["[", str(self.output_plus),"+/",str(self.output_minus),"-","]"]))
        else:
            print ("".join(["| " * (self.depth - 1), self.attribute, " = ", self.value, ": ", 
                           "[", str(self.output_plus),"+/",str(self.output_minus),"-","]"]))
        if self.children:
            for x in self.children.values():
                x.tree()
    


# In[ ]:


# Hard coded example, politician, education, music and cars
if 'example' in trainData:
    label = " Party "
    plus = "democrat"
    minus = "republican"
elif 'politician' in trainData:
    label = " Party "
    plus = "democrat"
    minus = "republican"
elif 'education' in trainData:
    label = "grade"
    plus = "A"
    minus = "notA"
elif 'music' in trainData:
    label = "hit"
    plus = "yes"
    minus = "no"
elif 'cars' in trainData:
    label = "class"
    plus = "yes"
    minus = "no"


# In[ ]:


decision_tree = tree(trainData, label = label, plus = plus, minus = minus)
decision_tree.result(testData)

