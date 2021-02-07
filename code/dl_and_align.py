import mmsdk
from mmsdk import mmdatasdk
import time
import numpy

def myavg(intervals,features):
	final=numpy.average(features,axis=0)
	if len(final.shape)==2:
		final=numpy.average(final,axis=0)
	return final

def align():
	#first time dl
	socialiq_no_align=mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel,"socialiq")
	#second time dl
	#socialiq_no_align=mmdatasdk.mmdataset("socialiq")
	#don't need these guys for aligning
	del socialiq_no_align.computational_sequences["SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE"]
	del socialiq_no_align.computational_sequences["SOCIAL-IQ_QA_BERT_MULTIPLE_CHOICE"]
	del socialiq_no_align.computational_sequences["SOCIAL_IQ_VGG_1FPS"]
	socialiq_no_align.align('SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT',collapse_functions=[myavg])
	#simple name change - now the dataset is aligned
	socialiq_aligned=socialiq_no_align
	
	socialiq_aligned.impute("SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT")
	socialiq_aligned.revert()
	
	deploy_files={x:x for x in socialiq_aligned.keys()}
	socialiq_aligned.deploy("./deployed",deploy_files)

if __name__=="__main__":
	align()
