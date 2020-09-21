A competition from Kaggle (https://www.kaggle.com/c/predicting-red-hat-business-value/overview) was selected where the company “Red Hat” has collected information related to behavior of their customers over a time and they wanted to predict which users they need to contact in future for selling their products. For this they require an algorithm to predict the customers who have potential business value for their company.

They provided 3 files (Persons.csv, act_test.csv, act_train.csv) of which we needed to merge two pair of files (Persons.csv & act_test.csv) & (Persons.csv & act_train.csv) to create a single unified data table.

The Persons file contained all of the unique peoples that have performed activities over time. Each row in the people file represents a unique person and each person has a unique people_id and contained activity_id column which represents unique activity performed by the user and also contained activities performed by user. And contains columns with char_* which represents particular characteristic for respective user.

For merging these files, I have used python and the respective code is as below. 

import pandas as pd
list1 = pd.read_csv('act_train.csv') 
list2 = pd.read_csv('people.csv')
res = pd.merge(list1, list2, on='people_id')
res.to_csv('data.csv', index=False)
list3 = pd.read_csv('act_test.csv') 
res1 = pd.merge(list3, list2, on='people_id') 
res1.to_csv('test_data.csv', index=False)

