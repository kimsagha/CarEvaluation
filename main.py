import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as f1s

global df


def read_data():
    print("Reading in data from car.data...")
    global df
    df = pd.read_csv('Data/car.data', sep=",", header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    print("Showing the first 5 data points")
    print(df.head())

    print("\nChecking data distribution\n(Figure 1)")
    class_names = df['class'].unique()
    class_counts = df['class'].value_counts()
    x_pos = np.arange(len(class_names))
    plt.bar(x_pos, class_counts, color=(0.5, 0.1, 0.5, 0.6))
    plt.title('Frequency of each class')
    plt.xlabel('classes')
    plt.ylabel('frequency')
    plt.xticks(x_pos, class_names)
    # plt.show()

    print('\nNumber of null values: ', df.isnull().sum().sum())

    # numerically encoding strings (discrete values) in df
    df['buying'] = df['buying'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3})
    df['maint'] = df['maint'].map({'vhigh': 0, 'high': 1, 'med': 2, 'low': 3})
    df['doors'] = df['doors'].map({'2': 0, '3': 1, '4': 2, '5more': 3})
    df['persons'] = df['persons'].map({'2': 0, '4': 1, 'more': 2})
    df['lug_boot'] = df['lug_boot'].map({'small': 0, 'med': 1, 'big': 2})
    df['safety'] = df['safety'].map({'low': 0, 'med': 1, 'high': 2})
    df['class'] = df['class'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})
    print('\nNumerically encoded dataset:\n', df.head())


def run_model():
    # split dataset into 70% training data and 30% test data (randomly)
    x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:6]], df['class'], test_size=0.3, random_state=0)

    # create and train classification model using logistic regression algorithm
    c_model = LogisticRegression(max_iter=500, random_state=0, C=5)
    c_model.fit(x_train, y_train)

    # run model
    y_pred = c_model.predict(x_test)

    # get model performance metrics
    score = c_model.score(x_test, y_test)
    print('\nAccuracy', score)

    f1_score = f1s(y_test, y_pred, average='micro')
    print('F1 score: ', f1_score)


read_data()
run_model()
