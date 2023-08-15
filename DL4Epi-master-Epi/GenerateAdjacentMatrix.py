import numpy as np
import pandas as pd

# generate a Matrix 15*15
# Matrix=np.ones((15,15)).astype(int)
Matrix=np.ones((29,29))
for i in range(0,len(Matrix)):
	Matrix[i][i]=0
# np.random.randint(0,2,size=[29,29])
print (Matrix)
Matrixdf= pd.DataFrame(Matrix)
Matrixdf.to_csv('./ind_mat2.txt', sep=',', index=False,header=False)