# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd


train = pd.read_csv("train.csv")

#%%

col_names = [f"var_{i}" for i in range(200)]
for col in col_names:
    #rr=train[col].apply(lambda x: True if train[col].value_counts()[x]==1 else False)
    #train[col + "_u"]=train[col].isin(rr)  # taks a looooot of time"""
    count = train[col].value_counts()
    uniques = count.index[count == 1]
    train[col + "_u"] = train[col].isin(uniques)
    #train[col + "_u"].apply(
        #lambda x: 1 if x is True else 0)
    
    #break

#%%
train["has_unique"] = train[[col +"_u" for col in col_names]].any(axis=1)

train=train.replace(True,1).replace(False,0)
#%%
train.to_csv("new_train.csv", index=False)