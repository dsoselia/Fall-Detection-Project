import os
from glob import glob

#current directory
top = os.getcwd() + '/CSV_fall_data/**/*.csv'

#takes path to csv, returns csv as string stripped of the metadata
def strip(path):
    with open(path) as csv:
        content = csv.readlines()
    stripped = ""
    i = 0
    for row in content:
        if (i > 5):
            stripped += row
        i+=1
    return stripped




#takes path with CSV files parent folder, merges into single CSV file. As default searches in current directory and outputs to merged.csv
def merge(path = os.getcwd() + '/CSV_fall_data/**/*.csv', mergedPath = 'merged.csv'):
    csvs = glob(top)
    merged = ""
    for csv in csvs:
        try:
            merged += strip(csv) + '\n'
        except:
            print('unsupported file')
    open(mergedPath, 'w').write(merged)
