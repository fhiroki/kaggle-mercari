
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import numpy as np


def display_all(df):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            display(df)


types_dict_train = {'train_id': 'int64', 'item_condition_id': 'int8',
                    'price': 'float64', 'shipping': 'int8'}
types_dict_test = {'test_id': 'int64', 'item_condition_id': 'int8',
                   'shipping': 'int8'}

train = pd.read_csv('train.tsv', delimiter='\t', low_memory=True,
                    dtype=types_dict_train)
test = pd.read_csv('test.tsv', delimiter='\t', low_memory=True,
                   dtype=types_dict_test)

train.head()
test.head()
train.shape, test.shape
display_all(train.describe(include='all').transpose())

train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')
train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')

test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')
test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')

train.dtypes, test.dtypes