import mmsdk
import os
import argparse
from mmsdk import mmdatasdk
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Reading dataset by files')
parser.add_argument('path', metavar='path', type=str, help='the folder path to read dataset from')
args = parser.parse_args()
dataset_dictionary={}


if(os.path.isdir(args.path) is False):
    print("Folder does not exist ...")
    exit(-1)

csdfiles = [f for f in listdir(args.path) if isfile(join(args.path, f)) and f[-4:]=='.csd']

if(len(csdfiles)==0):
    print("No csd files in thegiven folder")
    exit(-2)
print("%d csd files found"%len(csdfiles))

for csdfile in csdfiles:
    dataset_dictionary[csdfile] = os.path.join(args.path,csdfile)
    break
dataset=mmdatasdk.mmdataset(dataset_dictionary)

print ("List of the computational sequences")
print (dataset.computational_sequences.keys())
for key in dataset.computational_sequences.keys():
    value = dataset.computational_sequences[key]
    print(type(value))
    keys = list(value.keys())
    print(len(keys))
    print(value[keys[0]])
    
