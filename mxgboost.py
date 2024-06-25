import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Se emplean como features pm10..tempera
data = pd.read_excel('Calidad_del_Aire_Municipio_de_Duitama_20240623.xlsx')

features = data.iloc[:, 1:-1].values
target = data.iloc[:, -1].values


Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,target,test_size=0.2,random_state=2)
import xgboost as xgb
model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                              learning_rate=0.05, max_depth=4, 
                              min_child_weight=1.7817, n_estimators=2200,
                              reg_alpha=0.4640, reg_lambda=0.8571,
                               subsample=0.5213, silent=1,
                              random_state =42, nthread = -1)
model.fit(Xtrain,Ytrain)


Ypred = model.predict(Xtest)

# Calculate regression evaluation metrics
mae = mean_absolute_error(Ytest, Ypred)
mse = mean_squared_error(Ytest, Ypred)
r2 = r2_score(Ytest, Ypred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

import pickle
pickle.dump(model,open('co2_predic.pkl','wb'))
# con estos parametros las metrcias son: mae 25.6166, mse 1084, r2 0.94 con todos los datos 
