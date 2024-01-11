# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import subprocess
import pandas as pd
from pathlib import Path
import schedule
import time



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def execute_python_file(file_path):
   try:
      completed_process = subprocess.run(['python', file_path], capture_output=True, text=True)
      if completed_process.returncode == 0:
         print("Execution successful.")
         print("Output:")
         print(completed_process.stdout)
         print("execution finished")
      else:
         print(f"Error: Failed to execute '{file_path}'.")
         print("Error output:")
         print(completed_process.stderr)
   except FileNotFoundError:
      print(f"Error: The file '{file_path}' does not exist.")

def csv_path(csv):
    path = Path.cwd()
    path = str(path)
    path = path[:-4]
    path = path + csv
    f_path = Path(path)
    return f_path

def func():
    execute_python_file('models.py')
    print('executed')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print(Path.cwd())
    #execute_python_file('Data_analysis.py')
    #print("Data analysis done")
    #import Data_analysis as da
    print('input_data')
    #print(da.input_data)
    #print("__________________")
    #execute_python_file('drift detection.py')
    print("drift detection done")

    schedule.every(1).minutes.do(func)

    while True:
        schedule.run_pending()
        time.sleep(1)

    print(md.lr_results)
    print("--------------------------------------------")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
