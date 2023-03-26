import numpy as np
import pandas as pd

columns = [str(x) for x in range(401)]
df = pd.read_csv('img_data.csv',names=columns)
array_set = np.array(df)

x = array_set[:,:-1]
y = array_set[:,-1].reshape(-1,1)
for i in range(len(y)):
    if y[i,0] == 10:
        y[i,0] = 0
for i in range(len(x)):
    x[i] = (x[i].reshape(20,20).T).reshape(1,-1)

new = np.hstack((x,y))
df = pd.DataFrame(new,columns=columns)
df.to_csv('img_data_2.csv',header=False,index=False)



