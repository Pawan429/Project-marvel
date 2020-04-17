from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import json
import pandas as pd
import re, string
import nltk 
import lbsa

dir_path = "data/out_json/" 
csv_dir_path = "data/out_csv/"
Input_filename = 'all_srt.json'
Output_JSON_filename = 'sentiment.json'

input_filepath = dir_path + Input_filename
output_filepath = dir_path + Output_JSON_filename


def Sentiment_json_creator(Input_filename,Output_JSON_filename,csv_dir_path):
	raw_json_data =  open(Input_filename)
	json_data = json.load(raw_json_data)
	all_srt_SA_dict = {}
	for key in  list(json_data.keys()):
		movie_dict = json.loads(json_data[key])
		movie_dataframe = pd.DataFrame.from_dict(movie_dict)
		value, value_dataframe = sentiment_analysis_func(movie_dataframe)
		filename = str(key).split("/")[-1]
		csv_name = (csv_dir_path+ filename[:-4] +".csv") 
		value_dataframe.to_csv(csv_name, index = False)
		all_srt_SA_dict[key] = value
		print(csv_name, "created!")
	
	
	with open(Output_JSON_filename, 'w') as f:
		json.dump(all_srt_SA_dict, f, sort_keys=True, indent=4)
	
	print("Just created your new JSON file with SentiAnalysis results......:D")

	return

#Text Preprocessing functions


#remove_punctuation function
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

#remove whitespace from text function
def remove_whitespace(text): 
	return " ".join(text.split()) 

#remove stopwords function 
def lemmatization_and_stopwords(text): 
	stop_words = set(stopwords.words("english")) 
	word_tokens = word_tokenize(text) 
	filtered_text = [word for word in word_tokens if word not in stop_words] 
	lemmatizer = WordNetLemmatizer() 
	lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
	return lemmas 

#returns two outputs a json and a dataframe
def sentiment_analysis_func(movie_dataframe):
	#load the lexicon
	sa_lexicon = lbsa.get_lexicon('sa', language='english', source='nrc')
	movie_dataframe["sentiment_vector"]	= (
			movie_dataframe.text
			.str.lower()
			.map(remove_punctuation)
			.map(remove_whitespace)
			.map(lemmatization_and_stopwords)
			.map(" ".join)
			.map(lambda x : sa_lexicon.process(x)) #gets the sentiment vector
			)

	movie_dataframe["max_sentiment"] = movie_dataframe["sentiment_vector"].map(lambda x : max(x, key=x.get) if x[max(x, key=x.get)] >= 1 else "Neutral") #gets the key with max value
	json_data_SA = json.loads(movie_dataframe.to_json(orient = 'table'))
	return(json_data_SA,movie_dataframe) 



Sentiment_json_creator(input_filepath,output_filepath,csv_dir_path)
