import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3])], remainder='passthrough')
# X = ct.fit_transform(X).toarray()

transformer = pickle.load(open('transformer.sav', 'rb'))
regressor = pickle.load(open('modelforyield.sav', 'rb'))

Y_test = transformer.transform([['Andhra Pradesh',	'ANANTAPUR', 'Cotton(lint)', 'Kharif' , 1998 , 7300 , 2046.60877344]])

result = float(regressor.predict( Y_test))
print(type(result))
print(result)


