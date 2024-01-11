import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import mlflow
import mlflow.sklearn

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    print(path)
    path = path + csv
    f_path = Path(path)
    return f_path

input_data = pd.read_csv(csv_path('//data//healthcare_dataset.csv'))
print(input_data)
print(input_data.describe())
print(input_data.columns)
print('info', input_data.info())
print('shape', input_data.shape)
print('number of null value:', input_data.isnull().sum())
print('unique values of dependent variable', input_data['Test Results'].unique())

shape = input_data.shape
a = []
for i in input_data.columns:
    if(input_data[i].dtypes =='object'):
        a.append(i)
print('number of categorical variable',len(a))
for j in a:
    print('categorical variables: ',j)
    print('number of categories in',j,':',input_data[j].nunique())
    print('% of categories in',j,':',(input_data[j].value_counts()/shape[0])*100)
    print('==============================')

sns.countplot(x = input_data['Test Results'], hue = input_data['Gender'])
plt.show()
sns.countplot(x = input_data['Test Results'], hue = input_data['Blood Type'])
plt.show()

input_data = input_data.assign(Age_group=pd.cut(input_data['Age'], bins=[16, 30, 45, 90], right=False, labels=["young", "middle", "old"]))
print(input_data[["Age","Age_group"]].head(10))
input_data.drop(columns="Age",inplace=True)
sns.countplot(x = input_data['Test Results'], hue = input_data['Age_group'])
plt.show()
input_data.drop(columns=["Name","Room Number"],inplace=True)
print("value_counts of Hospital",input_data["Hospital"].value_counts())
print("unique values in Hospital",input_data["Hospital"].nunique())
print("value_counts of Doctor", input_data["Doctor"].value_counts())
print("unique values in Doctor", input_data["Doctor"].nunique())
#we can convert the dates into number of days in hospital
print("info",input_data.info())
input_data["Date of Admission"] = pd.to_datetime(input_data["Date of Admission"])
input_data["Discharge Date"] = pd.to_datetime(input_data["Discharge Date"])
input_data["days in Hospital"] = (input_data["Discharge Date"]-input_data["Date of Admission"]).dt.days
input_data.drop(columns=["Date of Admission","Discharge Date"], inplace=True)
print(input_data["days in Hospital"].describe())
print(input_data["Billing Amount"].describe())
input_data["Billing Amount"].hist()
plt.show()
plt.hist(input_data["Billing Amount"])
plt.show()
plt.boxplot(input_data["Billing Amount"])
plt.show()
input_data = input_data.assign(bill_group=pd.cut(input_data['Billing Amount'], bins=[0, 16666.666666666666666666666666667, 33333.333333333333333333333333334, 50000], right=False, labels=["less", "medium", "high"]))
sns.countplot(x = input_data['Test Results'], hue = input_data['bill_group'])
plt.show()
input_data.drop(columns="Billing Amount",inplace=True)
print(input_data["Medical Condition"].value_counts())
print(input_data)
#encoding input

def encoding(categorical,df):
    num_categorical = len(categorical)
    cat_dict = {}
    for i in categorical:
        lst = df[i].unique()
        list_length = len(lst)
        num = 0
        cat_val_inside = {}
        for j in lst:
            cat_val_inside[j]=num
            num+=1
        cat_dict[i]=cat_val_inside
    return cat_dict

categorical_cols = input_data.select_dtypes(include=['object','category']).columns
numberical_cols = input_data.select_dtypes(exclude='object').columns
categorical=[]
for cat in categorical_cols:
    categorical.append(cat)
numerical=[]
for num in numberical_cols:
    numerical.append(num)
encoded_data = input_data.replace(encoding(categorical,input_data))

print("encoded_data:",encoded_data)
encoded_data.drop(columns=["Doctor","Hospital"],inplace=True)
encoded_data[['Age_group', 'bill_group']] = encoded_data[['Age_group', 'bill_group']].astype('int64')
encoded_data.to_csv(csv_path('//data//meta_data(data_cleaned).csv'),index=False)
