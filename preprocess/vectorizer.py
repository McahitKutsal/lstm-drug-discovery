import numpy as np


def vectorize(smiles, embed, n_vocab):
    one_hot = np.zeros((smiles.shape[0], embed, n_vocab), dtype=np.int8)
    for i, smile in enumerate(smiles):
        one_hot[i,0,char_to_int["!"]] = 1
        for j, c in enumerate(smile):
            one_hot[i,j+1,char_to_int[c]] = 1
        one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
    return one_hot[:,0:-1,:], one_hot[:,1:,:]
