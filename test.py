df = pd.read_csv("../input/tpsfeb21-folds/train_folds.csv")
df_test = pd.read_csv("../input/tabular-playground-series-feb-2021/test.csv")
sample_submission = pd.read_csv("../input/tabular-playground-series-feb-2021/sample_submission.csv")

useful_features = [c for c in df.columns if c not in ("id", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
useful_features_test = [c for c in df.columns if c not in ("id", "target", "kfold")]
df_test = df_test[useful_features_test]

final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
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
    
    xtrain_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xtrain, label="target", task=tfdf.keras.Task.REGRESSION)
    xvalid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xvalid, label="target", task=tfdf.keras.Task.REGRESSION)
    xtest_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xtest, task=tfdf.keras.Task.REGRESSION)
    
    model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="better_default", task=tfdf.keras.Task.REGRESSION)
    model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x=xtrain_ds)
    
    preds_valid = model.predict(xvalid_ds)
    test_preds = model.predict(xtest_ds)
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

df = pd.read_csv("../input/tpsfeb21-folds/train_folds.csv")
df_test = pd.read_csv("../input/tabular-playground-series-feb-2021/test.csv")
sample_submission = pd.read_csv("../input/tabular-playground-series-feb-2021/sample_submission.csv")

useful_features = [c for c in df.columns if c not in ("id", "kfold")]
object_cols = [col for col in useful_features if 'cat' in col]
useful_features_test = [c for c in df.columns if c not in ("id", "target", "kfold")]
df_test = df_test[useful_features_test]

final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
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
    
    xtrain_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xtrain, label="target", task=tfdf.keras.Task.REGRESSION)
    xvalid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xvalid, label="target", task=tfdf.keras.Task.REGRESSION)
    xtest_ds = tfdf.keras.pd_dataframe_to_tf_dataset(xtest, task=tfdf.keras.Task.REGRESSION)
    
    model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)
    model.compile(metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x=xtrain_ds)
    
    preds_valid = model.predict(xvalid_ds)
    test_preds = model.predict(xtest_ds)
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
