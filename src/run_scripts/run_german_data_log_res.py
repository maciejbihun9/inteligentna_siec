
from src.data_manager import DataManager
from numpy import *
from src.norm_type import NormType
from src.normalizer import Normalizer
from src.log_res import LogRes

url = '../resources/german_data.txt'
data = DataManager.load_data(url, False, False)

categorical_mask = [True, False, True, True, False, True, True, False, True, True, False, True, False, True, True, False, True, False, True, True]

inputs = array([x[:19] for x in data])
target = array([y[20] for y in data])
target = array([0 if y == '2' else 1 for y in target])
# decode labeled data to numerical values
inputs = DataManager.categorize_data(inputs, categorical_mask)

# we can not normalize data that was hot encoded to numerical values
# here we assume that all data is already in numerical type
inputs = Normalizer.normalize(inputs.astype(float), NormType.stand_norm, [0, 1, 2, 3, 4, 5, 6])

X_train, X_test, y_train, y_test = DataManager.train_test_split(inputs, target, test_size=0.4, random_state=0)


log_res = LogRes(X_train, y_train, X_test, y_test)
results = log_res.adagrad_fit()
log_res.fit_test()
print(results)


"""
logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)
print(logreg.coef_)
"""