import pandas as pd 


with open('id2class_eurlex_subject_matter.txt','r') as file:
    lines=file.readlines()

# labels contains each label and doc id pair from the qrels in a separate list
labels=[]
for line in lines:
    labels.append(line.split()[0:2])


# labelsset contains only unique labels
labellist=[]
for label in labels:
    labellist.append(label[0])

labelset=set(labellist)

labelcount=[]
for label in labelset:
    labelcount.append([label,labellist.count(label)])

print(labelcount[1])