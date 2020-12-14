import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_parquet("data_test.parquet")

columns_ica = []
for col in data.columns:
	if 'ICA' in col:
		columns_ica.append(col)

X_ica = data[columns_ica].values

y = data.y.values


# with open('model_ica.pkl', 'rb') as f:
# 	model = pickle.load(f)

# y_predict = model.predict_proba(X_ica)[:,0]

model = GradientBoostingClassifier().fit(X_ica, y)
y_predict = model.predict_proba(X_ica)[:,1]


plt.plot(y_predict)
plt.plot(y)
plt.show()

with open('model_ica_2.pkl', 'wb') as f:
	pickle.dump(model, f)