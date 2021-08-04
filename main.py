import pandas as pd


def read_data():
    print("Reading in data from car.data")
    df = pd.read_csv('Data/car.data', sep=",")
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    print(df)


def check_data():
    print("checking data distribution and NaN values")


read_data()
