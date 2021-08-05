import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    print("Reading in data from car.data")
    df = pd.read_csv('Data/car.data', sep=",", header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    print("Showing the first 5 data points")
    print(df.head())

    print("\nChecking data distribution")
    class_names = df['class'].unique()
    class_counts = df['class'].value_counts()
    x_pos = np.arange(len(class_names))
    plt.bar(x_pos, class_counts, color=(0.5, 0.1, 0.5, 0.6))
    plt.title('Frequency of each class')
    plt.xlabel('classes')
    plt.ylabel('frequency')
    plt.xticks(x_pos, class_names)
    plt.show()

    print('Number of null values: ', df.isnull().sum().sum())


read_data()
