import csv
import features as ft
import pandas as pd


data1 = pd.read_csv('datasets/loan.csv')
data2 = pd.read_csv('datasets/borrower.csv')

output = pd.merge(data1,data2, on = 'date', how  = 'right')








    
