

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model, datasets, model_selection
import matplotlib.pyplot as plt
import numpy as np

from src.data_manager import DataManager

url = '../../resources/german_data.txt'
dataset = DataManager.load_data(url, False, True)

target = np.array([row[0] for row in dataset])
data = np.array([row[1:] for row in dataset])
# Amount, Country, TimeOfTransaction, BusinessType,
# NumberOfTransactionsAtThisShop, DayOfWeek
# które wartości są kategorialne, a któe ciągłe
categorical_mask = [True, False, True, True, False, True, True, False, True, True, False, True, False, True, True, False, True, False, True, True]

# categorical_mask = [False, True, True, True, False, True]
# pozwala na zmianę etykiety danej klasy na wartość liczbową
enc = LabelEncoder()
for i in range(0, data.shape[1]):
    # if it is a categorial value then replace it with value from encoder
    if (categorical_mask[i]):
        label_encoder = enc.fit(data[:, i])
        print ("Klasy kategorialne:", label_encoder.classes_)
        # change label classes on integer values
        integer_classes = label_encoder.transform(label_encoder.classes_)
        print("Klasy całkowito-liczbowe:", integer_classes)
        # refactor entire column using label encoder
        t = label_encoder.transform(data[:, i])
        # replace that column
        data[:, i] = t

# true values have items that are not categorical
mask = np.ones(data.shape, dtype=bool)
for i in range(0, data.shape[1]):
    if (categorical_mask[i]):
        mask[:, i] = False

# non categorical data
data_non_categoricals = data[:, np.all(mask, axis=0)]
# categorical data
data_categoricals = data[:, ~np.all(mask, axis=0)]
hotenc = OneHotEncoder()

hot_encoder = hotenc.fit(data_categoricals)
encoded_hot = hot_encoder.transform(data_categoricals)

new_data = np.append(data_non_categoricals, encoded_hot.todense(),1)
new_data = new_data.astype(np.float)

X_train, X_test, y_train, y_test = model_selection.train_test_split(new_data, target, test_size=0.4, random_state=0)
logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train,y_train)
log_output = logreg.predict_log_proba(X_test)
print("Szanse: "+ str(np.exp(logreg.coef_)))
print("Punkt przecięcia osi dla szans: " + str(np.exp(logreg.intercept_)))
print("Punkt przecięcia osi dla prawdopodobieństwa: " + str(np.exp(logreg.intercept_)/(1+np.exp(logreg.intercept_))))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.setp((ax1,ax2), xticks=[])

ax1.scatter(range(0,len(log_output[:,1]),1),
    log_output[:,1],
    s=100,
    label='Log. prawd.',color='Blue',alpha=0.5)

ax1.scatter(range(0,
    len(y_test),1),
    y_test,
    label='Wyniki',
    s=250,
    color='Green',
    alpha=0.5)

ax1.legend(bbox_to_anchor=(0., 1.02, 1., 0.102),
    ncol=2,
    loc=3,
    mode="expand",
    borderaxespad=0.)

ax1.set_xlabel('Przypadki testowe')
ax1.set_ylabel('Rzeczywiste wyniki / Log. prawd. wg modelu')
prob_output = [np.exp(x) for x in log_output[:,1]]

ax2.scatter(range(0,len(prob_output),1),
    prob_output,
    s=100,
    label='Prawd.',
    color='Blue',
    alpha=0.5)

ax2.scatter(range(0,len(y_test),1),
    y_test,
    label='Wyniki',
    s=250,
    color='Green',
    alpha=0.5)

ax2.legend(bbox_to_anchor=(0., 1.02, 1., 0.102),
    ncol=2,
    loc=3,
    mode="expand", borderaxespad=0.)

ax2.set_xlabel('Przypadki testowe')
ax2.set_ylabel('Rzeczywiste wyniki / Prawd. wg modelu')
plt.show()