
from src.data_manager import DataManager
from numpy import *
from src.norm_type import NormType
from src.normalizer import Normalizer
from src.log_res import LogRes

url = '../../resources/abalone.txt'
data = DataManager.load_data(url, False, False)
data = array(data)

inputs = data[:, 1:8]
inputs = inputs.astype(float)
target = data[:, 0]

# decode labeled data to numerical values
inputs = DataManager.categorize_data(target, [True])

# we can not normalize data that was hot encoded to numerical values
# here we assume that all data is already in numerical type
inputs = Normalizer.normalize(inputs.astype(float), NormType.stand_norm, [0, 1, 2, 3, 4, 5, 6])

X_train, X_test, y_train, y_test = DataManager.train_test_split(inputs, target, test_size=0.4, random_state=0)


log_res = LogRes(X_train, y_train, X_test, y_test)
results = log_res.fit_test()
print(results)


"""
logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)
print(logreg.coef_)
"""