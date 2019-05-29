import codecs
import arff
import pandas as pd
import pickle



file_ = codecs.open('C:/Users/Marcel Öfele/OneDrive/Dokumente/Studium/Master/2. Semester/ATiML/project/eurlex_tokenstring.arff','r','utf8')
data = arff.load(file_)

print(data.keys())

df=pd.DataFrame(data.get('data'))
df.to_pickle('data.pk1')
ddf = pd.read_pickle('data.pk1')