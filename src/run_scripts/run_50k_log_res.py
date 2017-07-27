
from src.data_manager import DataManager
from numpy import *
from src.norm_type import NormType
from src.normalizer import Normalizer
from src.log_res import LogRes

"""
In this tasks we lose information about index numbers of our columns.
In production we have to make sure that we insert new row properly converted.
That means we have to provide correct indexes number. We can do that 
"""

url = '../../resources/50k.txt'
data = DataManager.load_data(url, False, False)
data = array(data)

# filter
no_item_sign = '?'
data = DataManager.data_filter(data, no_item_sign)

categorical_mask = [False, True, False, True, False, True, True, True, True, True, False, False, False, True]

inputs = data[0:1000, 0:14]

# filter data

target = data[0:1000, 14]
target = array([0 if '<=50' in y else 1 for y in target])

# decode labeled data to numerical values
print("Categorizing...")
inputs = DataManager.categorize_data(inputs, categorical_mask)
print("Data has been categorized")

# we can not normalize data that was hot encoded to numerical values
# here we assume that all data is already in numerical type
print("Normalizing...")
inputs = Normalizer.normalize(inputs.astype(float), NormType.stand_norm, [0, 1, 2, 3, 4, 5])
print("Data has been normalized")

X_train, X_test, y_train, y_test = DataManager.train_test_split(inputs, target, test_size=0.4, random_state=0)

log_res = LogRes(X_train, y_train, X_test, y_test)
log_res.adagrad_fit()
results = log_res.fit_test()
print(results)


"""
logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)
print(logreg.coef_)
"""