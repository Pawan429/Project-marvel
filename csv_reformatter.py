import glob
import pandas as pd
import json

csv_dir_path = "data/out_csv/"

# csv = csv_list[1]
def csv_reformatter(csv_dir_path):
	for csv in glob.glob(csv_dir_path+"*.csv"):
		dw = pd.read_csv(csv)
		df = pd.concat([dw.drop(['sentiment_vector'], axis=1), dw['sentiment_vector'].apply(eval).apply(pd.Series)], axis=1)
		df.to_csv(csv)

	return(print("Sucessfully reformatted the sentiment_vector to 8 individual columns for all csvs in "))	

csv_reformatter(csv_dir_path)