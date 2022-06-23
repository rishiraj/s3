df = pd.read_csv("../input/30days-folds/train_folds.csv")
df_test = pd.read_csv("../input/30-days-of-ml/test.csv")
sample_submission = pd.read_csv("../input/30-days-of-ml/sample_submission.csv")

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
df_test = df_test[useful_features]

final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    valid_ids = xvalid.id.values.tolist()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
    
    params = {
        'random_state': 1, 
        'booster': 'gbtree',
        'n_estimators': 10000,
        'learning_rate': 0.03628302216953097,
        'reg_lambda': 0.0008746338866473539,
        'reg_alpha': 23.13181079976304,
        'subsample': 0.7875490025178415,
        'colsample_bytree': 0.11807135201147481,
        'max_depth': 3
    }
    
    model = XGBRegressor(
        n_jobs=4,
        **params
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)

print(np.mean(scores), np.std(scores))
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_1"]
final_valid_predictions.to_csv("train_pred_1.csv", index=False)

sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_1"]
sample_submission.to_csv("test_pred_1.csv", index=False)

----------------------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("../input/30days-folds/train_folds.csv")
df_test = pd.read_csv("../input/30-days-of-ml/test.csv")
sample_submission = pd.read_csv("../input/30-days-of-ml/sample_submission.csv")

useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
df_test = df_test[useful_features]

final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain =  df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()
    
    valid_ids = xvalid.id.values.tolist()

    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    xtrain[object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid[object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest[object_cols] = ordinal_encoder.transform(xtest[object_cols])
    
    params = {
        'learning_rate': 0.07853392035787837,
        'reg_lambda': 1.7549293092194938e-05,
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944, 
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3
    }
    
    model = XGBRegressor(
        random_state=fold,
        n_jobs=4,
        n_estimators=5000,
        **params
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(fold, rmse)
    scores.append(rmse)

print(np.mean(scores), np.std(scores))
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_2"]
final_valid_predictions.to_csv("train_pred_2.csv", index=False)

sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.columns = ["id", "pred_2"]
sample_submission.to_csv("test_pred_2.csv", index=False)
