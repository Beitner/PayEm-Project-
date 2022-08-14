from locale import currency
import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
from preprocesss import preprocess
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm
from catboost import Pool
from catboost.utils import get_roc_curve, select_threshold
from sklearn.metrics import ConfusionMatrixDisplay


import pickle
import numpy as np
import pandas as pd




def select_features(algorithm, steps, train_pool, validation_pool, features):
    """

    :param algorithm: algorithm used for calculating feature importance
    :param steps: number of iterations feature selection is performed
    :param train_pool: training set
    :param validation_pool: validation set
    :param features: from which feature algorithm will make a selection
    :return: dictionary containing features selected
    """
    print('Feature Selection Algorithm:', algorithm)
    model = CatBoostClassifier(iterations=500, random_seed=0)
    summary = model.select_features(
        train_pool,
        eval_set=validation_pool,
        features_for_select=features,
        num_features_to_select=10,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Verbose',
        # plot=True
    )
    print('Selected features:', summary['selected_features_names'])
    return summary


def train(train_pool):
    """

    :param train_pool: train set
    :return: trained catboost model

    """

    model = CatBoostClassifier(
        iterations=1000,
        verbose=10,
        custom_loss=['AUC']
    )
    model.fit(train_pool, plot=False)
    return model


def predict(df, model):
    """

    :param df: data to make predictions on
    :param model: trained model
    :return: array of predictions

    """
    predictions = model.predict(df)

    return predictions


def train1(data1):
    """

    :param df: data for training
    :return: model - trained catboost model
    :return: updated_categorical_features - categorical variables selected by algorithm during training
    :return: updated_numerical_features - numerical variables selected by algorithm during training
    :return: upper_threshold - threshold chosen by algorithm according to FPR from admin to make APPROVED prediction
    :return: lower_threshold - threshold chosen by algorithm according to FNR from admin to make DECLINED prediction
    :return: bertopic_model - trained bertopic model

    """

    data, bertopic_model = preprocess(data1)
    categorical_features = ['id', 'purchase_request_id', 'currency',
                            'categories', 'request_reason', 'occurance',
                            'request_type', '_accounting_id', '_department_id', '_sub_company_id',
                            'budget_item_id', 'send_budget_to_user_id', 'user_id', 'vendor_id', 'created_time_mon',
                            'created_time_day', 'created_time_week', 'invoice_attached', 'topic_num']
    numerical_features = ['amount']
    df = data[categorical_features + numerical_features]
    y = data['status']

    # split data
    X_train, X_validation, y_train, y_validation = train_test_split(df, y, test_size=0.2, random_state=0)
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categorical_features
    )

    validation_pool = Pool(
        data=X_validation,
        label=y_validation,
        cat_features=categorical_features
    )

    # select features
    shap_summary = select_features(algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues, steps=10,
                                   train_pool=train_pool, validation_pool=validation_pool,
                                   features=categorical_features + numerical_features)

    # update features selected
    updated_categorical_features = [feature for feature in categorical_features if
                                    feature in shap_summary['selected_features_names']]
    updated_numerical_features = [feature for feature in numerical_features if
                                  feature in shap_summary['selected_features_names']]

    # split data for final training
    df = data[updated_categorical_features + updated_numerical_features]
    y = data['status']

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=updated_categorical_features
    )

    # train the final model
    model = train(train_pool)
    y_pred = predict(X_test, model)
    probs = model.predict_proba(X_test)

    # select threshold with suits FPR specified by admin
    upper_threshold = select_threshold(model,
                                       train_pool,
                                       FPR=0.01)
    print('upper boundary:', upper_threshold)

    # select threshold with suits FNR specified by admin
    lower_threshold = select_threshold(model,
                                       train_pool,
                                       FNR=0.01)
    print('lower boundary:', lower_threshold)

    # take undecided out of prediction
    y_test_undecided = y_test[(probs[:, 1] < upper_threshold) & (probs[:, 1] > lower_threshold)]
    y_pred_undecided = probs[:, 1][(probs[:, 1] < upper_threshold) & (probs[:, 1] > lower_threshold)]

    # extract instances which will be classified
    y_test_classified = y_test[(probs[:, 1] > upper_threshold) | (probs[:, 1] < lower_threshold)]
    y_pred_classified = probs[:, 1][(probs[:, 1] > upper_threshold) | (probs[:, 1] < lower_threshold)]

    # make prediction upon thresholds
    y_pred_classified[y_pred_classified > upper_threshold] = 1
    y_pred = (y_pred_classified > lower_threshold).astype(int).astype(str)

    print(confusion_matrix(y_test_classified.to_numpy(), y_pred, labels=model.classes_))
    print(f'{len(y_pred_undecided)} requests require human intervention.')

    return (
    model, updated_categorical_features, updated_numerical_features, upper_threshold, lower_threshold, bertopic_model)

