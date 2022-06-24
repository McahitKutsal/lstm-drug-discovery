import numpy as np
from tensorflow.keras.utils import Sequence

class Data_Generator(Sequence):
    def __init__(self, input_data, labels, batch_size):
        self.input_data, self.labels = input_data, labels
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.input_data) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        x = self.input_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x, batch_y = np.array(x), np.array(y)
        
        return [batch_x, batch_x], batch_y