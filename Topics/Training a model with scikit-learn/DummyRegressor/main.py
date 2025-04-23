#  write your code here 
import pandas as pd
from sklearn.dummy import DummyRegressor

df = pd.read_csv(r'/Users/Macbook/PycharmProjects/Market Agents using FinRL/Topics/Training a model with scikit-learn/DummyRegressor/data/dataset/input.txt')

dummy_regressor = DummyRegressor(strategy='quantile', quantile=0.4)
dummy_regressor.fit(df['X'], df['y'])

answer = round(dummy_regressor.predict(df['X'])[0], 4)
print(answer)