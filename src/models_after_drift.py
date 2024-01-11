import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from warnings import filterwarnings
import winsound
import datetime

filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score, classification_report, \
    f1_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import mlflow
import mlflow.sklearn
import warnings
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema,ColSpec

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred, average):
    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred, average=average)
    precision = precision_score(actual, pred, average=average)
    f1 = f1_score(actual, pred, average=average)
    print(classification_report(actual, pred))
    return accuracy, recall, precision, f1


def model_building(model):
    m = model
    model = m.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    y_test_prob_1 = model.predict_proba(x_test)[:, 1]
    return model, y_test_pred, y_train_pred, y_test_prob_1


def model_hyperparameter_tuning(model, hyperparameter_dict, cv, n_jobs, scoring, error_score):
    ##cv = KFold(n_splits=10, n_repeats=3, random_state=1)
    model = GridSearchCV(estimator=model, param_grid=hyperparameter_dict, cv=cv, n_jobs=n_jobs, scoring=scoring,
                         error_score=error_score)
    model.fit(x_train, y_train)
    best = model.best_params_
    return best


def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    encoded_data = pd.read_csv(csv_path('//data//meta_data(data_cleaned)_after_drift_detection.csv'))
    x = encoded_data.drop(labels='Test Results', axis=1)
    y = encoded_data['Test Results']

    num = ['days in Hospital']

    ss = StandardScaler()
    x[num] = ss.fit_transform(x[num])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

    input_data = [
        {"name": "Gender", "type": "integer"},
        {"name": "Blood Type", "type": "integer"},
        {"name": "Medical Condition", "type": "integer"},
        {"name": "Insurance Provider", "type": "integer"},
        {"name": "Admission Type", "type": "integer"},
        {"name": "Medication", "type": "integer"},
        {"name": "Age_group", "type": "integer"},
        {"name": "days in Hospital", "type": "integer"},
        {"name": "bill_group", "type": "integer"}
    ]

    output_data = [{'type': 'integer'}]

    input_example = x_train.iloc[[0]]

    input_schema = Schema([ColSpec(col["type"], col['name']) for col in input_data])
    output_schema = Schema([ColSpec(col['type']) for col in output_data])

    mlflow.set_tracking_uri(uri="../mlruns")
    print("The set tracking uri is ", mlflow.get_tracking_uri())


    print("#################################Logistic#################################")

    exp = mlflow.set_experiment(experiment_name="Logistic after Datadrift")
    print('###########First Experiment Logistic after Datadrift #############')
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }
    print('##############first run###############')

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))
    mlflow.set_tags(tags)
    mlflow.autolog(
        log_input_examples=True
    )

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(lr)
    print(model)
    cm = confusion_matrix(y_test, y_test_pred)
    #sns.heatmap(cm, annot=True, fmt='.0f')
    #plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    lr_train_accuracy_score = accuracy
    lr_train_recall_score = recall
    lr_train_precision_score = precision
    lr_train_f1_score = f1
    print(" train accuracy_score: %s" % lr_train_accuracy_score)
    print(" train recall_score: %s" % lr_train_recall_score)
    print(" train precision_score: %s" % lr_train_precision_score)
    print(" train f1_score: %s" % lr_train_f1_score)

    k = KFold(n_splits=2, shuffle=True, random_state=10)
    scores = cross_val_score(lr, x, y, cv=k, scoring='f1_weighted')
    dt_bias = 1 - np.mean(scores)
    dt_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()


    print("######################Logistic Hyperparameter tuning######################")

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }
    print('#####################second run###############')

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    penalty = ['l1', 'l2', 'elasticnet']
    c_values = [0.1, 0.01, 0.001]
    tol = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    dual = [False, True]
    multi_class = ['auto', 'ovr', 'multinomial']
    warm_start = [False, True]
    n_job = [1, 2]
    grid = dict(solver=solvers, penalty=penalty, C=c_values, tol=tol, dual=dual, multi_class=multi_class,
                warm_start=warm_start, n_jobs=n_job)
    ##cv = KFold(n_splits=10, n_repeats=3, random_state=1)
    # lr = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
    # lr.fit(x_train,y_train)
    # best = model_hyperparameter_tuning(model,hyperparameter_dict, other)

    # best = lr.best_params_
    best = {'C': 1.0,
            'dual': False,
            'multi_class': 'auto',
            'n_jobs': 1,
            'penalty': 'l1',
            'solver': 'liblinear',
            'tol': 1,
            'warm_start': False}
    print('best logistic', best)

    lr = LogisticRegression(**best)
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(lr)
    print(model)

    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    lr_train_accuracy_score = accuracy
    lr_train_recall_score = recall
    lr_train_precision_score = precision
    lr_train_f1_score = f1
    print(" train accuracy_score: %s" % lr_train_accuracy_score)
    print(" train recall_score: %s" % lr_train_recall_score)
    print(" train precision_score: %s" % lr_train_precision_score)
    print(" train f1_score: %s" % lr_train_f1_score)

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(lr, x, y, cv=k, scoring='f1_weighted')
    lr_bias = 1 - np.mean(scores)
    lr_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(lr, "model", signature=signature, input_example=input_example)

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    print("#################################decision#################################")

    exp = mlflow.set_experiment(experiment_name="decision after Datadrift")

    print("###########second Experiment decision #############")
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }
    print("##############first run###############")

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))
    mlflow.set_tags(tags)
    mlflow.autolog(
        log_input_examples=True
    )

    dt = DecisionTreeClassifier(random_state=20)
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(dt)
    print(model)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    dt_train_accuracy_score = accuracy
    dt_train_recall_score = recall
    dt_train_precision_score = precision
    dt_train_f1_score = f1
    print(" train accuracy_score: %s" % dt_train_accuracy_score)
    print(" train recall_score: %s" % dt_train_recall_score)
    print(" train precision_score: %s" % dt_train_precision_score)
    print(" train f1_score: %s" % dt_train_f1_score)

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(dt, x, y, cv=k, scoring='f1_weighted')
    dt_bias = 1 - np.mean(scores)
    dt_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(dt, "model", signature=signature, input_example=input_example)

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()

    print("######################decision hyperparameter tuning######################")

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }
    print("#####################second run###############")

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    hyperparameter_dict = {'criterion': ['entropy', 'gini'], 'max_depth': np.arange(2, 15),
                           'min_samples_split': np.arange(2, 15)}

    best = model_hyperparameter_tuning(dt, hyperparameter_dict, cv=5, scoring='recall', n_jobs=None, error_score=np.nan)
    print(best)
    best = {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 6}
    dt = DecisionTreeClassifier(**best)
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(dt)
    print(model)

    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    dt_train_accuracy_score = accuracy
    dt_train_recall_score = recall
    dt_train_precision_score = precision
    dt_train_f1_score = f1
    print(" train accuracy_score: %s" % dt_train_accuracy_score)
    print(" train recall_score: %s" % dt_train_recall_score)
    print(" train precision_score: %s" % dt_train_precision_score)
    print(" train f1_score: %s" % dt_train_f1_score)

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(dt, x, y, cv=k, scoring='f1_weighted')
    dt_bias = 1 - np.mean(scores)
    dt_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(dt, "model", signature=signature, input_example=input_example)

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    print("###############################random Forest##############################")

    exp = mlflow.set_experiment(experiment_name="RandomForest after Datadrift")

    print("###########third Experiment random Forest #############")
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    print("##############first run###############")

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))
    mlflow.set_tags(tags)
    mlflow.autolog(
        log_input_examples=True
    )

    rf = RandomForestClassifier()
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(rf)
    print(model)
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    rf_train_accuracy_score = accuracy
    rf_train_recall_score = recall
    rf_train_precision_score = precision
    rf_train_f1_score = f1
    print(" train accuracy_score: %s" % rf_train_accuracy_score)
    print(" train recall_score: %s" % rf_train_recall_score)
    print(" train precision_score: %s" % rf_train_precision_score)
    print(" train f1_score: %s" % rf_train_f1_score)

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(rf, x, y, cv=k, scoring='f1_weighted')
    rf_bias = 1 - np.mean(scores)
    rf_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(rf, "model", signature=signature, input_example=input_example)

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()

    print("###################random Forest Hyperparameter tuning####################")

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }
    print("##############second run###############")

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    hyperparameter_dict = {'n_estimators': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy', 'log_loss'],
                           'max_depth': np.arange(2, 15), 'min_samples_split': [2, 5, 8, 10], 'random_state': [1, 2]}

    best = model_hyperparameter_tuning(rf, hyperparameter_dict, cv=None, scoring='recall', n_jobs=None,
                                       error_score=np.nan)
    print(best)
    best = {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 6}
    rf = RandomForestClassifier(**best)
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(rf)
    print(model)

    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='.0f')
    plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    rf_train_accuracy_score = accuracy
    rf_train_recall_score = recall
    rf_train_precision_score = precision
    rf_train_f1_score = f1
    print(" train accuracy_score: %s" % rf_train_accuracy_score)
    print(" train recall_score: %s" % rf_train_recall_score)
    print(" train precision_score: %s" % rf_train_precision_score)
    print(" train f1_score: %s" % rf_train_f1_score)

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(rf, x, y, cv=k, scoring='f1_weighted')
    rf_bias = 1 - np.mean(scores)
    rf_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(rf, "model", signature=signature, input_example=input_example)

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    import mlflow.xgboost
    import warnings
    from mlflow.models.signature import ModelSignature, infer_signature
    from mlflow.types.schema import Schema,ColSpec

    print("##################################xgboost#################################")

    exp = mlflow.set_experiment(experiment_name="Xgboost after Datadrift")

    print("###########forth Experiment xgboost #############")
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }
    print("##############first run###############")

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))
    mlflow.set_tags(tags)
    mlflow.xgboost.autolog(
        log_input_examples=True
    )

    xg = xgb.XGBClassifier()
    (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(xg)
    print(model)
    cm = confusion_matrix(y_test, y_test_pred)
    #sns.heatmap(cm, annot=True, fmt='.0f')
    #plt.show(block = False)
    (accuracy, recall, precision, f1) = eval_metrics(y_train, y_train_pred, None)
    xg_train_accuracy_score = accuracy
    xg_train_recall_score = recall
    xg_train_precision_score = precision
    xg_train_f1_score = f1
    print(" train accuracy_score: %s" % xg_train_accuracy_score)
    print(" train recall_score: %s" % xg_train_recall_score)
    print(" train precision_score: %s" % xg_train_precision_score)
    print(" train f1_score: %s" % xg_train_f1_score)
    mlflow.log_metric("training_accuracy_score", accuracy)
    mlflow.log_metric("recall", recall[0])
    mlflow.log_metric("precision", precision[0])
    mlflow.log_metric("f1", f1[0])

    k = KFold(n_splits=5, shuffle=True, random_state=48)
    scores = cross_val_score(xg, x, y, cv=k, scoring='f1_weighted')
    xg_bias = 1 - np.mean(scores)
    xg_var = np.std(scores) / np.mean(scores)
    print('Scores:', scores)
    print('Bias error:', 1 - np.mean(scores))
    print('Variance error:', np.std(scores) / np.mean(scores))

    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    mlflow.log_artifact(csv_path('//data//meta_data(data_cleaned).csv'))
    mlflow.sklearn.log_model(xg, "model", signature=signature, input_example=input_example)

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()

    # print("######################xgboost Hyperparameter tuning#######################")
    # mlflow.start_run(run_name="run2.1")
    # tags = {
    #     "engineering": "ML platform",
    #     "release.candidate": "RC1",
    #     "release.version": "2.0"
    # }
    #
    # mlflow.set_tags(tags)

    # current_run = mlflow.active_run()
    # print("Active run id is {}".format(current_run.info.run_id))
    # print("Active run name is {}".format(current_run.info.run_name))
    #
    # # hyperparameter_dict = {'n_estimators':[4,5,6,7,8],'criterion':['gini','entropy','log_loss'],
    # #     'max_depth':np.arange(2,15),'min_samples_split':[2, 5, 8, 10], 'random_state' : [1,2]}
    #
    # # best = model_hyperparameter_tuning(rf,hyperparameter_dict)
    # # print(best)
    # # best = {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 6}
    # # rf = RandomForestClassifier(**best)
    # # (model, y_test_pred, y_train_pred, y_test_prob_1) = model_building(rf)
    # # print(model)
    #
    # # cm = confusion_matrix(y_test,y_test_pred)
    # # sns.heatmap(cm,annot=True,fmt='.0f');
    # # plt.show(block = False)
    # # (accuracy, recall, precision, f1) = eval_metrics(y_train,y_train_pred, None)
    # # rf_train_accuracy_score = accuracy
    # # rf_train_recall_score = recall
    # # rf_train_precision_score = precision
    # # rf_train_f1_score = f1
    # # print(" train accuracy_score: %s" % rf_train_accuracy_score)
    # # print(" train recall_score: %s" % rf_train_recall_score)
    # # print(" train precision_score: %s" % rf_train_precision_score)
    # # print(" train f1_score: %s" % rf_train_f1_score)
    #
    # # k = KFold(n_splits = 5, shuffle = True, random_state = 48)
    # # scores = cross_val_score(rf,x,y,cv=k,scoring = 'f1_weighted')
    # # rf_bias = 1-np.mean(scores)
    # # rf_var = np.std(scores)/np.mean(scores)
    # # print('Scores:',scores)
    # # print('Bias error:',1-np.mean(scores))
    # # print('Variance error:', np.std(scores)/np.mean(scores))
    # artifacts_uri = mlflow.get_artifact_uri()
    # print("The artifact path is", artifacts_uri)
    # mlflow.end_run()

    result = pd.read_csv(csv_path('//data//result//result_accuracy.csv'))
    print(result)

    current = datetime.datetime.now()
    print(current)

    new_row = {'date_time': current, 'logistics': lr_train_accuracy_score, 'decision': dt_train_accuracy_score, 'random': rf_train_accuracy_score,'xgboost': xg_train_accuracy_score}
    print(new_row)
    result = result._append(new_row, ignore_index=True)
    result["date_time"] = pd.to_datetime(result["date_time"])
    print(result.info())
    result.to_csv(csv_path('//data//result//result_accuracy.csv'),index=False)

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

