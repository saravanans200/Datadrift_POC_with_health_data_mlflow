import pandas as pd
import numpy as np
import os
from pathlib import Path

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *

import datetime as dt
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path

original_dataset = pd.read_csv(csv_path('//data//healthcare_dataset.csv'))
print("old:", original_dataset)

updated_dataset = pd.read_csv(csv_path('//data//healthcare_dataset_updated.csv'))
print("updated",updated_dataset.tail(10))

insight = pd.read_csv(csv_path('//insights//insight_check.csv'))
insight

if updated_dataset.shape[0] != insight['no._of_rows'][0] or updated_dataset.shape[1] != insight['no._of_rows'][1]:
    print("some differnce")
    if updated_dataset['Age'].std() != insight['Age_std'][0]:
        print("evidently")
        print("send mail function here")
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        report.run(reference_data=original_dataset, current_data=updated_dataset)
        #report.save_html(r'C:\Users\A7202\Downloads\Datadrift_POC_with_health_data\artifact\result\drift_result.html')
        path = str(csv_path('//result//drift_result.html'))
        #report.save_html('C:/Users/A4647/OneDrive - Axtria/Desktop/Datadrift/Datadrift_POC_with_health_data/artifact/result/drift_result.html')
        report.save_html(path)
        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ])
        path = str(csv_path('//result//testsuite.html'))
        tests.save_html(path)
        tests.run(reference_data=original_dataset, current_data=updated_dataset)
        print("report uploaded")

else:
    print("no")

csv_path('//result//drift_result.html')
