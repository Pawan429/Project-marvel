import json
import pandas as pd


raw_json_data =  open('all_srt.json')

json_data = json.load(raw_json_data)

key = list(json_data.keys())[2]

movie_dict = json.loads(json_data[key])

movie_dataframe = pd.DataFrame.from_dict(movie_dict)

print(movie_dataframe.head())


from nltk.sentiment.vader import SentimentIntensityAnalyzer

polarity_extractor = SentimentIntensityAnalyzer()

movie_dataframe["sentiment"] = [(polarity_extractor.polarity_scores(x))["compound"] for x in movie_dataframe["text"]]
print(movie_dataframe.head())