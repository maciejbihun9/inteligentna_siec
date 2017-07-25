from src.data_manager import DataManager
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model, datasets, model_selection
import matplotlib.pyplot as plt
from numpy import *

class LogRes:

    """
    * Classifier has to be already trained
    * We can init our classifier with dataset or already trained weights:
    - we do not know the real number of weights in categorical systems
    - we can get them and init our classifier(But what if we miss one param?)
    * As input we can set weights or input data

    * I should test that classifier for three categorical/numerical datasets.
    * For now we can build our classifier using only file with categorical data.
    * set also which param we should tract as a categorical param.(In array of true/false bool values)
    * As a parameter we can set item to predict, dataset and also type of the data
    * There is no possible to build such classifier because target items not always are in 0, 1 values.
    * For this data we have to check is it 2 or 1
    * The best what you can do is just doing everything in the same time.
    * do not waste performance.
    * Method should return 0 or 1.
    """
    @staticmethod
    def classify(data_array: ndarray, data_item: ndarray, categorical_mask: list):
        """

        :param data_array: Data to train our classifier. Target is always in the first column
        :param data_item: This is data_item sa ndarray to predict. Contains one column less that data_array
        :param data_cate: list of data categories
        :return: int value that determines how to
        """
        # copy much needed data from run file

        # divide data to target and data
        # data = array([x[:19] for x in data_array])
        # target = array([y[20] for y in data_array])

        data = array([x[:19] for x in data_array])
        target = array([y[20] for y in data_array])

        target = [0 if y == '2' else 1 for y in target]

        # replace categorical data with new integer values that represents lables.
        enc = LabelEncoder()
        for i in range(0, data.shape[1]):
            if (categorical_mask[i]):
                label_encoder = enc.fit(data[:, i])
                print("Klasy kategorialne:", label_encoder.classes_)
                integer_classes = label_encoder.transform(label_encoder.classes_)
                print("Klasy całkowito-liczbowe:", integer_classes)
                t = label_encoder.transform(data[:, i])
                data[:, i] = t

        mask = ones(data.shape, dtype=bool)
        for i in range(0, data.shape[1]):
            if (categorical_mask[i]):
                mask[:, i] = False

        # non categorical data
        data_non_categoricals = data[:, all(mask, axis=0)]

        # categorical data
        data_categoricals = data[:, ~all(mask, axis=0)]
        hotenc = OneHotEncoder()

        hot_encoder = hotenc.fit(data_categoricals)
        encoded_hot = hot_encoder.transform(data_categoricals)

        # combine data_categoricals with non_data_categoricals where data_categoricals have new hot values
        new_data = append(data_non_categoricals, encoded_hot.todense(), 1)
        new_data = new_data.astype(float)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(new_data, target, test_size=0.4,
                                                                            random_state=0)
        logreg = linear_model.LogisticRegression(tol=1e-10)
        logreg.fit(X_train, y_train)
        log_output = logreg.predict_log_proba(X_test)
        print("Logarith output : {}".format(log_output))
        print("Szanse: {}".format(str(exp(logreg.coef_))))
        print("Punkt przecięcia osi dla szans: {}".format(str(exp(logreg.intercept_))))
        print("Punkt przecięcia osi dla prawdopodobieństwa: {}".format(str(exp(logreg.intercept_) / (1 + exp(logreg.intercept_)))))
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        plt.setp((ax1, ax2), xticks=[])

        ax1.scatter(range(0, len(log_output[:, 1]), 1),
                    log_output[:, 1],
                    s=100,
                    label='Log. prawd.', color='Blue', alpha=0.5)

        ax1.scatter(range(0,
                          len(y_test), 1),
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
        prob_output = [exp(x) for x in log_output[:, 1]]

        ax2.scatter(range(0, len(prob_output), 1),
                    prob_output,
                    s=100,
                    label='Prawd.',
                    color='Blue',
                    alpha=0.5)

        ax2.scatter(range(0, len(y_test), 1),
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


data_item = array([0, 22.67, 7, 2, 8, 4, 0.165, 0, 0, 0, 0, 2, 160, 1, 0])
url = '../resources/german_data.txt'
data_list = DataManager.load_data(url, False, False)
# categories are only for data not targets
categories = [True,False,True ,True ,False, True, True, False, True, True, False, True, False, True, True ,False ,True ,False ,True ,True]
result = LogResClass.classify(data_list, data_item, categories)