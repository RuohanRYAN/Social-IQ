import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy
import torch.optim as optim
import time
import scipy.misc
import os
from model import mylstm
import numpy
import pickle
from random import shuffle
import time
import numpy as np
from collections import OrderedDict
from torch.distributions.multinomial import Multinomial
from transformers import PreTrainedTokenizer
from transformers import BertTokenizer,BertModel
import copy

def main(path,dest):
	for filename in os.listdir(path):
		qa_path = path+filename
		f = open(qa_path,"r")
		tk = BertTokenizer.from_pretrained("bert-base-uncased")
		max_len = 0
		model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
		
		text_array = []
		PlaceHolder = []
		qa = []
		qi = []
		for line in f:
			line_array = line.split(":")
			text = line_array[1]
			ID = line_array[0]

			if(ID[0]=="q"):
				PlaceHolder = []
				PlaceHolder.append(clean_text(text))
			else:
				if(ID[0]=="a"):
					cp = copy.deepcopy(PlaceHolder)
					cp.append(clean_text(text))
					qa.append(cp)
				else:
					cp = copy.deepcopy(PlaceHolder)
					cp.append(clean_text(text))
					qi.append(cp)
			# print(ID)
			text_array.append(text)
		tokens = tk(qa, add_special_tokens=True, padding="max_length", max_length = 64,return_token_type_ids=True, return_attention_mask=True,truncation=True)
		tokens_i = tk(qi, add_special_tokens=True, padding="max_length", max_length = 64,return_token_type_ids=True, return_attention_mask=True,truncation=True)



		decoded_strings = tk.batch_decode(tokens["input_ids"])

		### look at the results ###
		for i in range(len(decoded_strings)):
			print(decoded_strings[i])
			print(tokens["input_ids"][i])
			print(tokens["token_type_ids"][i])
			print(tokens["attention_mask"][i])
			print(len(tokens["input_ids"][i]))
			break

		####
		ave_token_a = get_tokens(tokens,model)
		ave_token_i = get_tokens(tokens_i,model)
		# print(ave_token_a.shape)
		# print(ave_token_i.shape)

		directory = filename.split(".")[0]
		os.mkdir(os.path.join(dest,directory))
		np.save(dest+directory+"/a.npy", ave_token_a.numpy())	
		np.save(dest+directory+"/i.npy", ave_token_i.numpy())
		print("------------------------")
		# break


		# break



def clean_text(text):
	return text.lower()[:-2].strip()

def get_tokens(tokens,model):
	model.eval()
	input_ids = torch.tensor(tokens["input_ids"], dtype=torch.long)
	input_type = torch.tensor(tokens["token_type_ids"], dtype= torch.long)
	input_attention = torch.tensor(tokens["attention_mask"], dtype = torch.long)

	with torch.no_grad():
		output = model(input_ids=input_ids,attention_mask = input_attention,token_type_ids=input_type)

		hidden_states = output[2]
		# print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
		# layer_i = 0
		# print ("Number of batches:", len(hidden_states[layer_i]))
		# batch_i = 0
		# print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
		# token_i = 0
		# print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

		token_embedding = torch.stack(hidden_states, dim=0)
		# print(token_embedding.shape)
		# ave_token = torch.mean(token_embedding,2)
		ave_token = torch.mean(token_embedding,0)

		# print(ave_token.shape)
		# print(token_embedding.size())

	return ave_token


qa_path = "/home/gaoruohan19/project/Social-IQ/data/rawdata/raw/qa/"
dest = "/home/gaoruohan19/project/Social-IQ-new/data/qai_updated/"
main(qa_path,dest)