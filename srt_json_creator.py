import pandas as pd
import os, glob
import datetime
import json


#loading all srt files from data/Input_SRT_directory

input_dir_path = "data/Input_SRT_directory/"
output_dir_path = "data/out_json/"
srt_list = [file for file in glob.glob(str(input_dir_path)+"*.srt")]


def timestring_to_timestamp(timestring):	
	# converting the sting to a datetime object
	start = datetime.datetime.strptime(timestring, "%H:%M:%S,%f")
	#evaluating seconds value
	timestamp = ((start.hour * 60 + start.minute) * 60 + start.second) 
	return(timestamp)


def dialogue_dataframe_extractor(filename):

	file = open(filename,'r')

	#make a list of lines
	lines = [line for line in file.readlines()]
	lines = [line.replace('\ufeff', '') for line in lines]

	# remove the /n part at the last for each element
	lines = [i[:-1] for i in lines]
	lines = lines[lines.index(str(1)):]

	scenes = []
	Scene_count = 1
	while True:
		print("working...")
		try:	
			j = lines.index(str(Scene_count+1))			
			# print(j)
			scenes.append(list(lines[lines.index(str(Scene_count)):j]))
			Scene_count += 1
		except:
			break
	scenes.append(lines[lines.index(str(Scene_count)):])


	scenes = [scene[:-1] for scene in scenes]

	scenes = [[scene[0],scene[1]," ".join(scene[2:])] for scene in scenes]

	# Make the DataFrame
	srt_dataframe = pd.DataFrame(scenes, columns=["Scene", "timestamp", "text"])
	#extract start time from time_stamp
	srt_dataframe["start_time"] = srt_dataframe['timestamp'].str[:12]
	# convert starttime to seconds
	srt_dataframe["secs_stamp"] = [timestring_to_timestamp(i) for i in srt_dataframe["start_time"]]
	# drop starttime column
	srt_dataframe = srt_dataframe[["secs_stamp","text"]]
	return(srt_dataframe)


def srt_df_dict_creator(srt_list):
	srt_dict = {}
	for filename in srt_list:
		print(filename)
		try:	
			srt_dict[filename] = dialogue_dataframe_extractor(filename).to_json()
		except:	
			pass
	
	return(srt_dict)



all_srt_json = srt_df_dict_creator(srt_list)	

print(all_srt_json.keys())


#save dict to json file

with open(str(output_dir_path) +'all_srt.json', 'w') as fp:
    json.dump(all_srt_json, fp)



