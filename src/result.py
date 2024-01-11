import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    print(path)
    path = path + csv
    f_path = Path(path)
    return f_path

result_data = pd.read_csv(csv_path('//data//result//result_accuracy.csv'))
print(result_data)


plt.plot(result_data.date_time, result_data.logistics)

plt.plot(result_data.date_time, result_data.decision, color='green')

plt.plot(result_data.date_time, result_data.random, color='red')

plt.plot(result_data.date_time, result_data.xgboost, color='yellow')

# Giving title to the graph
plt.title('model testing')

# rotating the x-axis tick labels at 30degree
# towards right
plt.xticks(rotation=30, ha='right')

# Giving x and y label to the graph
plt.xlabel('Date')
plt.ylabel('models')
plt.show()
