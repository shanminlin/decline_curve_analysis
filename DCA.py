import numpy as np
import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=DataConversionWarning)
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from lmfit import Model
from math import sqrt
from contextlib import contextmanager


@contextmanager
def timer(title):
    """Computes time taken for each operation."""
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def load(file_path):
    """loads data"""
    data_raw = pd.read_csv(file_path)
    return data_raw


def convert_date(train_raw, test_raw):
    """Converts the 'DAILY_RDG_DATE' column to the number of days from start date for each well.

    Args:
        train_raw - Pandas dataframe of training data.
        test_raw - Pandas dataframe of testing data.

    Returns:
        minmax_trainval - Pandas dataframe of training data with added column 'OPERATING_DAYS'
        test - Pandas dataframe of testing data with added column 'OPERATING_DAYS'
    """

    traintest = pd.concat([train_raw, test_raw], sort=True)

    # Convert the data type of 'DAILY_RDG_DATE' to datetime for further manipulation
    traintest['DAILY_RDG_DATE'] = pd.to_datetime(traintest['DAILY_RDG_DATE'])

    # Convert 'DAILY_RDG_DATE' to 'OPERATING_DAYS' for each well
    traintest['STARTING_DATE'] = traintest.groupby('WELL_NUM')['DAILY_RDG_DATE'].transform('min')  # find start date for each well
    traintest['OPERATING_DAYS'] = (traintest['DAILY_RDG_DATE'] - traintest['STARTING_DATE']).dt.days
    traintest = traintest.drop(['STARTING_DATE'], axis=1)

    train = traintest.iloc[:len(train_raw), :]
    test = traintest.iloc[len(train_raw):, :]

    return train, test


def scale(train, test):
    """Fits a scaler that transforms each specified feature in train set to a range between 0 and 1 for each well,
    and applies the scaler to transform test set.

    Args:
        train - Pandas dataframe of training data.
        test - Pandas dataframe of testing data.

    Returns:
        minmax_train - Pandas dataframe of scaled training data.
        minmax_test - Pandas dataframe of scaled testing data.
    """

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()  # default=(0, 1)

    test = test.fillna(0)

    minmax_train_list = []
    minmax_test_list = []

    # Compute a list of well numbers in train and test
    well_list_train = train['WELL_NUM'].unique().tolist()
    well_list_test = test['WELL_NUM'].unique().tolist()

    for well in well_list_train:
        # split data by well number
        train_well = train.loc[train.WELL_NUM == well]

        # fit scaler
        scaler_well = scaler.fit(train_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']])

        # transform features in train
        minmax_train_well = pd.DataFrame(data=train_well)
        minmax_train_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']] = scaler_well.transform(
            train_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']])
        minmax_train_list.append(minmax_train_well)

        # transform features in test
        if well in well_list_test:
            test_well = test.loc[test.WELL_NUM == well]
            minmax_test_well = pd.DataFrame(data=test_well)
            minmax_test_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']] = scaler_well.transform(
                test_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']])
            minmax_test_list.append(minmax_test_well)

    minmax_trainval = pd.concat(minmax_train_list)
    minmax_test = pd.concat(minmax_test_list)

    return minmax_trainval, minmax_test


def train_val_split(minmax_trainval):
    """splits training data into training and validation set

    Args:
        minmax_trainval - Pandas dataframe of training data.

    Returns:
        minmax_train - Pandas dataframe of training data (80%).
        minmax_val - Pandas dataframe of validation data (20%).
    """

    minmax_train_list = []
    minmax_val_list = []

    for well in minmax_trainval['WELL_NUM'].unique().tolist():
        # split data by well number and sort rows by increasing operating days
        minmax_trainval_well = minmax_trainval.loc[minmax_trainval.WELL_NUM == well]
        minmax_trainval_well = minmax_trainval_well.sort_values(by=['OPERATING_DAYS'])

        # for each well, split data into 80% training and 20% validation set
        train = minmax_trainval_well.iloc[:int(len(minmax_trainval_well) * 0.8), :]
        val = minmax_trainval_well.iloc[int(len(minmax_trainval_well) * 0.8):, :]

        minmax_train_list.append(train)
        minmax_val_list.append(val)

    minmax_train = pd.concat(minmax_train_list)
    minmax_val = pd.concat(minmax_val_list)

    return minmax_train, minmax_val


def tag_outliers(minmax_train, oil_or_gas):
    """Tags outliers in the data.

    Outliers are labeled with '-1' while normal data are labeled with '1'.

    Args:
        minmax_train - Pandas dataframe of training data.
        (str) oil_or_gas - 'oil' or 'gas', indicating whether we want to retrieve oil or gas production data.

    Returns:
        minmax_train_tag - Pandas dataframe of training data with an additional column
        'Outlier_Tag' indicating whether each row is an outlier for oil production or gas production.
    """

    # set outlier fraction
    outlier_fraction = 0.5  # fraction can be set between 0 (exclusive) and 0.5 (inclusive)

    # define the outlier detection method
    algorithm = IsolationForest(behaviour='new', contamination=outlier_fraction, random_state=42)

    outlier_tag = np.array([])

    for well in minmax_train['WELL_NUM'].unique().tolist():
        minmax_train_well = minmax_train.loc[minmax_train.WELL_NUM == well]

        # keep only the rate and time columns
        if oil_or_gas == 'oil':
            train_well = minmax_train_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS']]
        else:
            train_well = minmax_train_well[['OPERATING_DAYS', 'GROSS_GAS_MCF']]

        # for each well, fit the data and tag outliers
        outlier_tag_well = algorithm.fit(train_well).predict(train_well)
        outlier_tag = np.append(outlier_tag, outlier_tag_well)

    minmax_train['Outlier_Tag'] = outlier_tag
    minmax_train_tag = pd.DataFrame(data=minmax_train)

    return minmax_train_tag


def remove_outliers(minmax_train_tag, oil_or_gas):
    """Remove rows that are tagged as outliers in the data, and save outliers to a csv file.

    Outliers are labeled with '-1' while normal data are labeled with '1'.

    Args:
        minmax_train_tag - Pandas dataframe of training data with outliers indicated

    Returns:
        minmax_train_without_outlier - Pandas dataframe of training data with outliers in oil or gas production
        removed.
    """

    minmax_train_without_outlier = minmax_train_tag[minmax_train_tag['Outlier_Tag'] != -1]
    minmax_train_outlier = minmax_train_tag[minmax_train_tag['Outlier_Tag'] == -1]
    minmax_train_outlier.to_csv('outlier in {}'.format(oil_or_gas))

    return minmax_train_without_outlier


def remove_improper_wells(minmax_train_without_outlier, oil_or_gas):
    """Removes wells that have fewer than 30 data points.

    Args:
        minmax_train_without_outlier - Pandas dataframe of training data.
        (str) oil_or_gas - 'oil' or 'gas', indicating whether we want to retrieve oil or gas production data.

    Returns:
        minmax_train_without_outlier - Pandas dataframe of training data with wells that have fewer than 30 data points
        removed.
    """
    well_list = minmax_train_without_outlier['WELL_NUM'].unique().tolist()
    improper_well_list = []
    for i, well in enumerate(well_list):
        train_well = minmax_train_without_outlier[minmax_train_without_outlier.WELL_NUM == well]
        if train_well['WELL_NUM'].value_counts().item() < 30:
            print('{} does not have sufficient training data.'.format(well))
            improper_well_list.append(well)

    minmax_train_without_outlier = minmax_train_without_outlier[
        ~minmax_train_without_outlier.WELL_NUM.isin(improper_well_list)]

    return minmax_train_without_outlier


def exp_func(x, qi, di):
    """Evaluates exponential function with parameters.

    Args:
        x - array of x values (days)
        qi - initial flow rate
        di - initial decline rate

    Returns:
        NumPy array of calculated y values
    """
    return qi * np.exp(-di * x)


def har_func(x, qi, di):
    """Evaluates harmonic function with parameters.

     Args:
        x - array of x values (days)
        qi - initial flow rate
        di - initial decline rate

    Returns:
        NumPy array of calculated y values
    """
    return qi * (1 + di * x) ** (-1)


def hyp_func(x, qi, b, di):
    """Evaluates hyperbolic function with parameters.

     Args:
        x - array of x values (days)
        qi - initial flow rate
        di - initial decline rate
        b - exponent, between 0 and 2

    Returns:
        NumPy array of calculated y values
    """
    return qi * (1 + b * di * x) ** (-1 / b)


def hyp2exp_func(x, qi, di, b, df):
    """Evaluates hyperbolic function with parameters.

     Args:
        x - array of x values (days)
        qi - initial flow rate
        di - initial decline rate
        b - exponent, between 0 and 2
        df - transition decline rate

    Returns:
        q - NumPy array of calculated y values
    """
    # set conditions for when transition occurs (x_trans)
    x_trans = (di / df - 1) / (b * di)

    # first computes hyperbolic curves
    q_trans = hyp_func(x_trans, qi, b, di)
    q = hyp_func(x, qi, b, di)

    # computes exponential curves after transition point (x_trans)
    q_exp = exp_func(q_trans, df, x[x > x_trans])
    q[x > x_trans] = q_exp

    return q


def curve_fit(minmax_train_without_outlier, oil_or_gas):
    """Fit four decline curves to each well for oil and gas production.

    Args:
        minmax_train_without_outlier - Pandas dataframe of training data.
        (str) oil_or_gas - 'oil' or 'gas', indicating whether we want to retrieve oil or gas production data.

    Returns:
        train_result - list of lmfit ModelResult objects of four decline curves for oil or gas production for each well
    """

    # define curve fitting functions to be compared
    fitting_functions = [('Exponential', exp_func),
                         ('Harmonic', har_func),
                         ('Hyperbolic', hyp_func),
                         ('Hyperbolic-to-Exponential', hyp2exp_func)]

    train_result = []

    well_list = minmax_train_without_outlier['WELL_NUM'].unique().tolist()
    for i, well in enumerate(well_list):
        train_well = minmax_train_without_outlier[minmax_train_without_outlier.WELL_NUM == well]
        # get production rate as y and time as x
        x_train_well = train_well['OPERATING_DAYS']

        if oil_or_gas == 'oil':
            y_train_well = train_well['GROSS_OIL_BBLS']
        else:
            y_train_well = train_well['GROSS_GAS_MCF']

        result_well_list = []
        for name, function in fitting_functions:
            if name == 'Hyperbolic':
                model = Model(function)
                model.set_param_hint('qi', value=0.1, min=0)
                model.set_param_hint('di', value=0.1, min=0)
                model.set_param_hint('b', value=0.1, min=0, max=2)

            elif name == 'Hyperbolic-to-Exponential':
                model = Model(function)
                model.set_param_hint('qi', value=0.3, min=0)
                model.set_param_hint('di', value=0.3, min=0)
                model.set_param_hint('b', value=0.3, min=0, max=2)
                model.set_param_hint('df', value=0.1, vary=False)
                model.set_param_hint('x_trans', value=0.3, min=0)

            else:
                model = Model(function)
                model.set_param_hint('qi', value=0.1)
                model.set_param_hint('di', value=0.1)

            result_well = model.fit(y_train_well, x=x_train_well)
            result_well_list.append(result_well)

        train_result.append(result_well_list)

    return train_result


def evaluate_fit(train_result, minmax_train_without_outlier, minmax_val, oil_or_gas):
    """Evalutes fitting of four decline curves on validation set.

    Args:
        train_result - list of lmfit ModelResult objects of four decline curves for oil or gas production for each well.
        minmax_train_without_outlier - Pandas dataframe of processed data used for training.
        minmax_val - Pandas dataframe of validation set.
        (str) oil_or_gas - 'oil' or 'gas', indicating whether we want to retrieve oil or gas production data.

    Returns:
        pred_RMSE - Pandas dataframe of root-mean-square-error for oil or gas production for each well.
    """

    well_list = minmax_train_without_outlier['WELL_NUM'].unique().tolist()

    pred_RMSE = []
    for i, well in enumerate(well_list):
        well = minmax_val[minmax_val.WELL_NUM == well]
        x_well = well['OPERATING_DAYS']

        if oil_or_gas == 'oil':
            y_well = well['GROSS_OIL_BBLS']
        else:
            y_well = well['GROSS_GAS_MCF']

        # predict for oil production rate
        pred_RMSE_well = []
        for j in range(len(train_result[i])):
            pred_well = train_result[i][j].eval(x=x_well)
            pred_rmse_well = sqrt(mean_squared_error(y_well, pred_well))
            pred_RMSE_well.append(pred_rmse_well)

        pred_RMSE.append(pred_RMSE_well)

    # convert RMSE results to Pandas dataframe
    pred_RMSE = pd.DataFrame(pred_RMSE, columns=['exponential', 'harmonic', 'hyperbolic', 'hyperbolic-to-exponential'],
                             index=well_list)

    pred_RMSE.to_csv('RMSE_on_validation_set.csv')

    return pred_RMSE


def select_fit(pred_RMSE):
    """Selects the best fitting functions based on RMSE on validation set.

    Args:
        pred_RMSE -Pandas dataframe of root-mean-square-error for each well.

    Returns:
        pred_RMSE_best - Pandas dataframe of best fitting functions for each well.
    """
    pred_RMSE['lowest_RMSE'] = pred_RMSE.idxmin(axis=1)

    # select the best prediction
    pred_RMSE_best = pred_RMSE['lowest_RMSE']
    pred_RMSE_best = pred_RMSE_best.to_frame()

    return pred_RMSE_best


def predict(train_result, minmax_test, pred_RMSE_best):
    """Evalutes fitting of four decline curves to each well on validation set.

    Args:
        train_result - list of lmfit ModelResult objects of four decline curves for oil or gas production for each well.
        minmax_test - Pandas dataframe of test set.
        pred_RMSE_best - Pandas dataframe of best fitting functions for each well.

    Returns:
        pred_test - Pandas dataframe of normalized predictions for test set for each well.
    """

    well_list = minmax_test['WELL_NUM'].unique().tolist()

    pred_test = pd.DataFrame()

    for i, well in enumerate(well_list):
        well = minmax_test[minmax_test.WELL_NUM == well]
        x_well = well['OPERATING_DAYS']
        train_result_well = train_result[i]

        fit_function = pred_RMSE_best.iloc[i, :]['lowest_RMSE']
        if fit_function == 'exponential':
            pred_well = train_result_well[0].eval(x=x_well)
        elif fit_function == 'harmonic':
            pred_well = train_result_well[1].eval(x=x_well)
        elif fit_function == 'hyperbolic':
            pred_well = train_result_well[2].eval(x=x_well)
        else:
            pred_well = train_result_well[3].eval(x=x_well)

        well['prediction'] = pred_well
        pred_test = pd.concat([pred_test, well])

    return pred_test


def combine_pred_test(pred_test_oil, pred_test_gas):
    """Combined normalized predictions on test set for both oil and gas production.

   Args:
       pred_test_oil - Pandas dataframe of normalized predictions for oil production rate for test set.
       pred_test_gas - Pandas dataframe of normalized predictions for gas production rate for test set.

   Returns:
       pred_test_combined - Pandas dataframe of normalized predictions for both oil and gas production rate for test set .
   """


    pred_test_oil = pred_test_oil[['WELL_NUM', 'DAILY_RDG_DATE', 'OPERATING_DAYS', 'prediction']]
    pred_test_gas = pred_test_gas[['prediction']]

    pred_test_oil = pred_test_oil.rename(columns={"prediction": "GROSS_OIL_BBLS"})
    pred_test_gas = pred_test_gas.rename(columns={"prediction": "GROSS_GAS_MCF"})

    pred_test_combined = pd.concat([pred_test_oil, pred_test_gas], axis=1)

    return pred_test_combined


def inverse_transform(train, pred_test):
    """Transforms back each specified feature in data to original scale for each well.

    Args:
        train - Pandas dataframe of training data that is used to fit the scaler
        pred_test - Pandas dataframe of predictions for test set.

    Returns:
        pred_test_transformed - Pandas dataframe of predictons transformed to original scales for test set.
    """

    scaler = MinMaxScaler()

    # Compute a list of well numbers in train and test
    well_list_test = pred_test['WELL_NUM'].unique().tolist()

    pred_test_transformed_list = []

    for well in well_list_test:
        # split data by well number
        train_well = train.loc[train.WELL_NUM == well]
        pred_test_well = pred_test.loc[pred_test.WELL_NUM == well]

        # fit scaler
        scaler_well = scaler.fit(train_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']])

        # transform features in train
        pred_test_transformed_well = pd.DataFrame(data=pred_test_well)
        pred_test_transformed_well[
            ['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']] = scaler_well.inverse_transform(
            pred_test_well[['OPERATING_DAYS', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']])

        pred_test_transformed_list.append(pred_test_transformed_well)

    pred_test_transformed = pd.concat(pred_test_transformed_list)
    pred_test_transformed['GROSS_OIL_BBLS'][pred_test_transformed['GROSS_OIL_BBLS'] < 0] = 0
    pred_test_transformed['GROSS_GAS_MCF'][pred_test_transformed['GROSS_GAS_MCF'] < 0] = 0
    pred_test_transformed[['WELL_NUM', 'DAILY_RDG_DATE', 'GROSS_OIL_BBLS', 'GROSS_GAS_MCF']].to_csv('submission_vv.csv',
                                                                                                    index=False)

    return pred_test_transformed


def main():
    with timer("Load train & test "):
        train_raw = load('train_ops_results.csv')
        test_raw = load('test_ops_results_blank.csv')
    with timer("Convert date for train & test"):
        train, test = convert_date(train_raw, test_raw)
    with timer("Scale train & test"):
        minmax_trainval, minmax_test = scale(train, test)
    with timer("Split train to train and validation"):
        minmax_train, minmax_val = train_val_split(minmax_trainval)
    with timer("Tag outliers in train for oil and gas production"):
        minmax_train_tag_oil = tag_outliers(minmax_train, 'oil')
        minmax_train_tag_gas = tag_outliers(minmax_train, 'gas')
    with timer("Remove outliers in train for oil and gas production"):
        minmax_train_without_outlier_oil = remove_outliers(minmax_train_tag_oil, 'oil')
        minmax_train_without_outlier_gas = remove_outliers(minmax_train_tag_gas, 'gas')
    with timer("Remove wells with few data in train for oil and gas production"):
        minmax_train_without_outlier_oil = remove_improper_wells(minmax_train_without_outlier_oil, 'oil')
        minmax_train_without_outlier_gas = remove_improper_wells(minmax_train_without_outlier_gas, 'gas')
    with timer("Fit curves for oil and gas production"):
        train_result_oil = curve_fit(minmax_train_without_outlier_oil, 'oil')
        train_result_gas = curve_fit(minmax_train_without_outlier_gas, 'gas')
    with timer("Evaluate fit for oil and gas production"):
        pred_RMSE_oil = evaluate_fit(train_result_oil, minmax_train_without_outlier_oil, minmax_val, 'oil')
        pred_RMSE_gas = evaluate_fit(train_result_gas, minmax_train_without_outlier_gas, minmax_val, 'gas')
    with timer("Select best fit for oil and gas production"):
        pred_RMSE_best_gas = select_fit(pred_RMSE_gas)
        pred_RMSE_best_oil = select_fit(pred_RMSE_oil)
    with timer("Predict for test data"):
        pred_test_oil = predict(train_result_oil, minmax_test, pred_RMSE_best_oil)
        pred_test_gas = predict(train_result_gas, minmax_test, pred_RMSE_best_gas)
        pred_test_combined = combine_pred_test(pred_test_oil, pred_test_gas)
    with timer("Transform prediction to original scale"):
        pred_test_transformed = inverse_transform(train, pred_test_combined)

if __name__ == "__main__":
    with timer("Full model run"):
        main()