# kaggle-video-game-play

A kaggle competition solution code

This competition is the individual project of MSBD 5001. The individual project counts for 20% of the final grade.

## Description

Jeremy is a core video game player. He has collected more than 400 video games with different types and played more than 100 of them. Your task is to predict how many hours Jeremy has spent on each game, given some of the game features including the purchased time, price, release date, popular user-defined tags, etc. 80% of the dataset is provided as the training data including playing time, the rest are used as testing data, where only features are provided. You have to submit the predicted results of these testing samples, which are then compared with the ground truth to evaluate the performance of your model.

## Evaluation Metric
The evaluation metric is the root mean-squared error on the testing dataset in hours. Lower error leads to a higher ranking.

## Grading
The individual project has 20 scores in total. If you submit the result and the code is available on Github, you get at least 10. The other ten scores are “ranking points”. Sorted by ranking, top-1 will get full score (10), while the last get 0 out of 10. The obtained score is uniformly distributed across ranks.

## File descriptions
* **train.csv** - the training set, with the target label column provided named "playtime_forever" in hours.
* **test.csv** - the test set, with only features
* **sampleSubmission.csv** - a sample submission file in the correct format

## Data fields
* **is_free** - Whether the game is free.
* **price** -The price of the game.
* **genres** - Genres of the game.
* **categories** - Categories of the game according to the video game digital distribution platform.
* **tags** - Popular user-defined tags.
* **purchase_date** - The date the game was purchased by Jemery.
* **release_date** - The release date of the game.
* **total_positive_reviews** - The total number of positive reviews received by the game.
* **total_negative_reviews** -The total number of negative reviews received by the game.

## Tips

To handle some features provided, extra functions are needed. Some useful links are provided here.

Handle string in Pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html

Get dummies from strings with separator: https://www.geeksforgeeks.org/python-pandas-series-str-get_dummies/

Handle date format in Pandas: https://medium.com/datadriveninvestor/how-to-work-with-dates-in-pandas-like-a-pro-a84055a4819d
