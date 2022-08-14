def prediction(model_trained, features1, features2, upper_threshold, lower_threshold, bertopic_model, Request_amount,
               Currency, Expense_date, Expense_category, Sub_company, Request_title, Request_decription):
    import pandas as pd
    import numpy as np
    import random
    from preprocesss import preprocess
    from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm
    from catboost import Pool

    data = pd.read_csv('data_translated.csv', parse_dates=['created'])
    data_size = data.shape
    row = random.randrange(data_size[0])

    data.loc[(row), "amount"] = Request_amount
    data.loc[(row), "created"] = Expense_date + ".898024 UTC"
    if Expense_category != "":
        data.loc[(row), "categories"] = Expense_category

    comp_name = data[data["api_company_name"] == Sub_company].api_company_id
    if comp_name.shape[0] > 0:
        data.loc[(row), "api_company_id"] = (np.unique(comp_name.values))
        data.loc[(row), "api_company_name"] = Sub_company
    else:
        data.loc[(row), "api_company_id"] = 999
        data.loc[(row), "api_company_name"] = Sub_company

    data.loc[(row), "title_Eng"] = Request_title
    data.loc[(row), "request_reason_Eng"] = Request_decription

    print(data.iloc[(row), :])
    data12 = preprocess(data, betropic_model=bertopic_model)

    df = data12[features1 + features2]
    y = data12['status']

    train_pool = Pool(
        data=df,
        label=y,
        cat_features=features1
    )

    y_th = model_trained.predict_proba(df)

    if y_th[row,1] >= upper_threshold:
        print('APPROVED')
    elif y_th[row,1] <= lower_threshold:
        print("DECLINED")
    else:
        print("Requires manual approval.")

    print("probs:")
    print(y_th[row])


    return
