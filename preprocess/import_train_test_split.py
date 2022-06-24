import pandas as pd
from preprocess.vectorizer import vectorize
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/HIV.csv')

df['smiles'] = df['smiles'].astype('str')
mask = (df['HIV_active']==1)
df = df.loc[mask]
mask = (df['smiles'].str.len() < 90)
df = df.loc[mask]
smiles = df['smiles']
data = df['smiles']
full_train, test = train_test_split(data, test_size=0.2, random_state=17)

val_split = 0.10
train, val_set = train_test_split(full_train, test_size=val_split, random_state=17)

embed = 100
n_vocab = 54

X_train, y_train = vectorize(train, embed, n_vocab)
X_val, y_val = vectorize(val_set, embed, n_vocab)
X_test, y_test = vectorize(test, embed, n_vocab)