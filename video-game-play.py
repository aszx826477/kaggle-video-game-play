# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

np.set_printoptions(suppress=True)


# %%
# Read csv file into dataframe
train_df = pd.read_csv("train.csv", parse_dates=["purchase_date", "release_date"])
test_df = pd.read_csv("test.csv", parse_dates=["purchase_date", "release_date"], index_col = 0)

# %% [markdown]
# ## Part 1 Data processing
# * Do one-hot encoding for `tags`, `genres` and `categories` of trainning dataset and test dataset respectively. Then union the one-hot encoding of and drop duplicates of columns. 
# * Split the date into year and month ignoring day. 
# * Add a new feature *interval* = *purchase_date_day* - *release_date_day*. 

# %%
# Add a feature of interval using purchase_date - release_date
train_df["interval"] = train_df["purchase_date"].apply(lambda x: x.day) - train_df["release_date"].apply(lambda x: x.day)

def extract_date(df, column):
    df[column + "_year"] = df[column].apply(lambda x: x.year)
    df[column + "_month"] = df[column].apply(lambda x: x.month)

extract_date(train_df, "purchase_date")
extract_date(train_df, "release_date")

train_df = train_df.fillna(0.0)


# %%
# One-hot encoding for trainning data
tags_train = train_df["tags"].str.get_dummies(",") # shape = (357, 312)
genres_train = train_df["genres"].str.get_dummies(",") # (357, 20)
categories_train = train_df["categories"].str.get_dummies(",") # (357, 29)

diff = genres_train.columns.difference(tags_train.columns) # drop duplicates of columns
train_one_hot = tags_train.join(genres_train[diff]) # (357, 312)
 
diff = categories_train.columns.difference(train_one_hot.columns)
train_one_hot = train_one_hot.join(categories_train[diff]) # (357, 340)


# %%
# One-hot endcoding for test data
tags_test = test_df["tags"].str.get_dummies(",") # (90, 229)
genres_test = test_df["genres"].str.get_dummies(",") # (90, 14)
categories_test = test_df["categories"].str.get_dummies(",") # (90, 28)

diff = genres_test.columns.difference(tags_test.columns)
test_one_hot = tags_test.join(genres_test[diff]) # (90, 229)

diff = categories_test.columns.difference(test_one_hot.columns)
test_one_hot = test_one_hot.join(categories_test[diff]) # (90, 256)


# %%
# Union train_one_hot and test_one hot features
# Fill the NaN with 0.0

diff = test_one_hot.columns.difference(train_one_hot.columns) # (5,)
train_one_hot = pd.concat([train_one_hot, pd.DataFrame(columns=list(diff))], axis=1) # (357, 345)
train_one_hot = train_one_hot.fillna(0.0)

diff = train_one_hot.columns.difference(test_one_hot.columns) # (89,)
test_one_hot = pd.concat([test_one_hot, pd.DataFrame(columns=list(diff))], axis=1) # (90, 345)
test_one_hot = test_one_hot.fillna(0.0)

# %% [markdown]
# ## Part 2 Prepare data for trainning
# 
# * `[X_train, y_train]` use to train a basic model
# * `[X_train_correct, y_train_corretc]` use to train a correction model

# %%
# The full trainning dataset. 
# Use it to train and predict on test dataset at first.
X_train = train_df.join(train_one_hot)                   .drop(["genres", "tags", "categories", "purchase_date", "release_date",                          "id", "playtime_forever"], axis = 1)
y_train = train_df["playtime_forever"]


# %%
# The sub-dataset (playtime_forever > 3) of the oringinal trainning set. 
# Use the sub-dataset to re-train a new model and then do prediction again to correct the outliers.
train_df_correct = train_df[train_df["playtime_forever"] > 3]
X_train_correct = train_df_correct.join(train_one_hot)                                   .drop(["genres", "tags", "categories", "purchase_date","release_date",                                          "id", "playtime_forever"], axis = 1)
y_train_correct = train_df_correct["playtime_forever"]

# %% [markdown]
# ## Part 3 Train models

# %%
# Train a decision tree regression model using full trainning dateset
regDT = DecisionTreeRegressor(random_state=0)
regDT.fit(X_train, y_train)


# %%
# Train a correction decision tree regression model using sub trainning dateset
regDTCorrect = DecisionTreeRegressor(random_state=0)
regDTCorrect.fit(X_train_correct, y_train_correct)


# %%
# Train a ramdom forest tree regression model using full trainning dataset
regRF = RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
regRF.fit(X_train, y_train)  

# %% [markdown]
# ## Part 4 Predict on test dataset

# %%
# Read test.csv again
test_df = pd.read_csv("test.csv", parse_dates=["purchase_date", "release_date"], index_col = 0)


# %%
# Pre-processing the test data
test_df["interval"] = test_df["purchase_date"].apply(lambda x: x.day) - test_df["release_date"].apply(lambda x: x.day)

extract_date(test_df, "purchase_date")
extract_date(test_df, "release_date")

test_df = test_df.fillna(0.0)


# %%
# prepare the test data for prediction
X_test = test_df.join(test_one_hot).drop(["genres", "tags", "categories", "purchase_date", "release_date"], axis = 1)


# %%
regDT.predict(X_test)


# %%
regDTCorrect.predict(X_test)


# %%
regRF.predict(X_test)


# %%
correct_distance = regDTCorrect.predict(X_test) - regDT.predict(X_test) # calculate the correction distance
filter = np.vectorize(lambda x: 0 if x < 30 else x) # filter the distance that < 30
correct_vector = filter(correct_distance)
correct_vector

# %% [markdown]
# ## Part 5 Generate submission csv file

# %%
# First submission: desicion tree + correct vector
y_test_predict = regDT.predict(X_test) + correct_vector
result = pd.DataFrame({"id":range(90), "playtime_forever":y_test_predict})
result.to_csv("submission_DT_correct.csv", index=False)

# Before add the correct vector, the DT model achieves 11+ score in public board.
# After using correction, it can achieve 4+ score in public board.
# However, the TA warned that the high performance in public board didn't mean the model also performed well in private board.


# %%
# Second submission: random forest
y_test_predict = regRF.predict(X_test)
result = pd.DataFrame({"id":range(90), "playtime_forever":y_test_predict})
result.to_csv("submission_RF.csv", index=False)

# The RF model gaines 15+ score in public board.

