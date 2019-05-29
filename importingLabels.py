import pandas as pd 


with open('id2class_eurlex_subject_matter.txt','r') as file:
    lines=file.readlines()

# labels contains each label and doc id pair from the qrels in a separate list
labels=[]
for line in lines:
    labels.append(line.split()[0:2])

print(labels[0])

# labelsset contains only unique labels
labelset=[]
for label in labels:
    labelset.append(label[0])

labelsset=set(labelset)

