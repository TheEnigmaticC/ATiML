import codecs
import arff
import pandas as pd
import pickle



file_ = codecs.open('C:/Users/marti/Documents/mter/clases/2o semestre/ATiML/project/eurlex_token.arff','r','utf8')
data = arff.load(file_)

print(data.keys())

df=pd.DataFrame(data.get('data'))
df.to_pickle('data.pk1')
ddf = pd.read_pickle('data.pk1')