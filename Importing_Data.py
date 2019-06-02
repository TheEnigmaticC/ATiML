import codecs
import arff
import pandas as pd
import pickle



file_ = codecs.open('C:/Users/JuliaUser/Documents/EurLex/eurlex_tokenstring.arff/eurlex_tokenstring.arff/eurlex_tokenstring.arff','r','utf8')
data = arff.load(file_)

print(data.keys())

df=pd.DataFrame(data.get('data'))
df.to_pickle('data.pkl')
ddf = pd.read_pickle('data.pkl')
