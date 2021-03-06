import numpy as np
import pandas as pd

class MySimpleScaler(object):
 

 def preprocess(self, data):
    
    #data = data[data.weight_pounds > 0]
    data = data[data.mother_age  > 0]
    data = data[data.plurality > 0]
    data = data[data.gestation_weeks > 0]
    print(data.shape)

    
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(data['is_male'])
    # Drop column B as it is now encoded
    data = data.drop('is_male',axis = 1)
    # Join the encoded df
    data = data.join(one_hot)


    return data
