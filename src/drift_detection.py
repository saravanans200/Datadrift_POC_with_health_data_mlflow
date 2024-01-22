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

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

smtp_server = "smtp.gmail.com"
smtp_port = 587  # TLS port for Gmail
smtp_username = "saravanashanmuganathan35@gmail.com"  # Your Gmail email address
smtp_password = "ngld jpvc mszi kejt"  # Your generated Gmail App Password

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    #path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path

def send_email(subject, body, to_address, attachments=None):
    from_email = smtp_username

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_address
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    if attachments:
        for file_path in attachments:
            filename = Path(file_path).name
            attachment = open(csv_path(file_path), "rb")
            p = MIMEBase('application', 'octet-stream')
            p.set_payload((attachment).read())
            encoders.encode_base64(p)
            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
            msg.attach(p)

    text = msg.as_string()

    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Use TLS for encryption
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(from_email, to_address, text)
        server.quit()

        return {"message": "Email sent successfully!"}
    except Exception as e:
        print(f"Exception when sending email: {str(e)}")
        return {"message": "Email sending failed."}

original_dataset = pd.read_csv(csv_path('//data//healthcare_dataset.csv'))
print("old:", original_dataset)

updated_dataset = pd.read_csv(csv_path('//data//healthcare_dataset_updated.csv'))
print("updated",updated_dataset.tail(10))

insight = pd.read_csv(csv_path('//insights//insight_check.csv'))

if updated_dataset.shape[0] != insight['no._of_rows'][0] or updated_dataset.shape[1] != insight['no._of_rows'][1]:
    print("some differnce")
    if updated_dataset['Age'].std() != insight['Age_std'][0]:
        print("evidently")

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

        content = '''Hi Team, 
                      Data Drift has happened, please check the data. There is a change in data.
        Thank you'''

        subject = "drift detection found"
        to_address = ["saravana.shanmuganathan@axtria.com"]
        attachments = ["//result//drift_result.html", "//result//accuracy_dashboard.html"]

        print("Sending mail...")
        for i in to_address:
            # Send the email and store the response
            #email_response = send_email(subject, content, i, attachments)  #commenting email function. test is working
            # Print the status of the email sending process
            #print(email_response)
            print("Mail sent to " + i)


        print("report uploaded")

else:
    print("no")

csv_path('//result//drift_result.html')
