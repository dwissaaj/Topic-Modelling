import pandas as pd
encoding = 'unicode_escape'
list_data = pd.read_csv("tweet.csv", encoding= 'unicode_escape', sep='delimiter', header=None,engine='python')
list_data['teks'] = list_data
data1 = list_data.drop(labels=0,axis=1)
data1 = data1.to_json('/home/levi/PycharmProjects/file.json')

