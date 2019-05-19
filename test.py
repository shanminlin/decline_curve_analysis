import DCA
import pytest
import numpy as np
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=DataConversionWarning)

train = None
test = None
train_date = None
test_date = None

minmax_trainval = None
minmax_train = None
minmax_val = None
minmax_test = None
minmax_train = None

minmax_train_tag_oil = None
minmax_train_without_outlier_oil = None
minmax_train_without_outlier_oil_all = None

def setup_module(module):
    global train
    global test
    global train_date
    global test_date
    global minmax_trainval
    global minmax_train
    global minmax_val
    global minmax_test
    global minmax_train_tag_oil
    global minmax_train_without_outlier_oil
    global minmax_train_without_outlier_oil_all
    global minmax_train_without_outlier
    train = DCA.load('train_sample_for_unit_test.csv')
    test = DCA.load('train_sample_for_unit_test.csv')

    train_date, test_date = DCA.convert_date(train, test)
    minmax_trainval, minmax_test = DCA.scale(train_date, test_date)

    minmax_train, minmax_val = DCA.train_val_split(minmax_trainval)
    minmax_train_tag_oil = DCA.tag_outliers(minmax_train, 'oil')
    minmax_train_without_outlier_oil = DCA.remove_outliers(minmax_train_tag_oil, 'oil')
    minmax_train_without_outlier_oil_all = DCA.remove_improper_wells(minmax_train_without_outlier_oil, 'oil')

def test_load():
    assert train.shape[1] == 7


def test_convert_date():
    assert train_date.shape[1] == 8
    assert 'OPERATING_DAYS' in train_date.columns


def test_scale():
    assert round(minmax_trainval['GROSS_GAS_MCF'].min(), 1) == 0.0
    assert round(minmax_trainval['OPERATING_DAYS'].min(), 1) == 0.0
    assert round(minmax_trainval['GROSS_OIL_BBLS'].min(), 1) == 0.0
    assert round(minmax_trainval['GROSS_GAS_MCF'].max(), 1) == 1.0
    assert round(minmax_trainval['OPERATING_DAYS'].max(), 1) == 1.0
    assert round(minmax_trainval['GROSS_OIL_BBLS'].max(), 1) == 1.0


def test_train_val_split():
    assert len(minmax_train) + len(minmax_val) == len(minmax_trainval)


def test_tag_outliers():
    assert 'Outlier_Tag' in minmax_train_tag_oil.columns


def test_remove_outliers():
    assert minmax_train_without_outlier_oil['Outlier_Tag'][minmax_train_without_outlier_oil['Outlier_Tag'] == 1].all()


def test_remove_improper_wells():
    assert minmax_train_without_outlier_oil_all.shape[1] == 9