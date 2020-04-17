import glob
import pandas as pd
import json

csv_list = [file for file in glob.glob("*.csv")]
# csv = csv_list[1]
for csv in csv_list:
	dw = pd.read_csv(csv)
	df = pd.concat([dw.drop(['sentiment_vector'], axis=1), dw['sentiment_vector'].apply(eval).apply(pd.Series)], axis=1)
	df.to_csv(csv)
