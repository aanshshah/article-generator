import torch
import torch.tensor
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from tqdm import tqdm

class headLM(nn.Module):
	def __init__(self,vocab_sz,word_embed_sz,hidden_size,sentence_len):
		super(headLM,self).__init__()
		
		self.embed_words = nn.Embedding(vocab_sz, word_embed_sz)
		self.decoder = nn.GRU(word_embed_sz,hidden_size,num_layers=2,batch_first=True)
		self.hidden2vocab = nn.Linear(hidden_size,vocab_sz)
		self.dropout = nn.Dropout(0.25) 


	def forward(self,batched_inp,batched_labels,padding=-1,h0=None,use_drop_out=False):	
		batch_size = batched_inp.size(0)
		embedded = self.embed_words(batched_inp)
		if use_drop_out:
			embedded = self.dropout(embedded)
		out_states,next_state = self.decoder(embedded,h0)
		logits = self.hidden2vocab(out_states)
		mask = (batched_labels != padding).double()
		# print(mask)
		# print(batched_inp)
		# print(batched_labels)
		loss = self.sequence_cross_entropy_with_logits_or_prob(logits, batched_labels, mask)
		return loss,next_state.data

	def gen(self):
		sentence = [0]
		prev_state = None
		while sentence[-1] != 2 and len(sentence) < 20: 
			embedded = self.embed_words(torch.LongTensor([[sentence[-1]]])) 
			out_states,prev_state= self.decoder(embedded,prev_state)
			logits = self.hidden2vocab(out_states).data.numpy()[0][0]
			prbs = np.exp(logits)/np.sum(np.exp(logits))
			topk = sorted(list(np.arange(prbs.shape[0])),key=lambda x: prbs[x],reverse=True)[0:20]
			
			topkprbs = list(map(lambda x: prbs[x],topk))
			# print(topk)
			# print(topkprbs)
			new_word = np.random.choice(topk,p=topkprbs/np.sum(topkprbs))

			sentence.append(new_word)
		return sentence

	def sequence_cross_entropy_with_logits_or_prob(self,logits,targets,weights,label_smoothing=None,takeLogits=True):
	    """
	    Returns
	    -------
	    A torch.FloatTensor representing the cross entropy loss.
	    If ``batch_average == True``, the returned loss is a scalar.
	    If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).
	    """
	    logits_flat,log_probs_flat = None,None
	    if takeLogits:
	        # shape : (batch * sequence_length, num_classes)
	        logits_flat = logits.view(-1, logits.size(-1))
	        # shape : (batch * sequence_length, num_classes)
	        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
	        # shape : (batch * max_len, 1)
	    else:
	        log_probs_flat = torch.log(logits.view(-1,logits.size(-1)))

	    targets_flat = targets.contiguous().view(-1, 1).long()

	    if label_smoothing is not None and label_smoothing > 0.0:
	        num_classes = logits.size(-1)
	        smoothing_value = label_smoothing / num_classes
	        # Fill all the correct indices with 1 - smoothing value.
	        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
	        smoothed_targets = one_hot_targets + smoothing_value
	        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
	        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
	    else:
	        # Contribution to the negative log likelihood only comes from the exact indices
	        # of the targets, as the target distributions are one-hot. Here we use torch.gather
	        # to extract the indices of the num_classes dimension which contribute to the loss.
	        # shape : (batch * sequence_length, 1)
	        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
	    # shape : (batch, sequence_length)
	    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
	    # shape : (batch, sequence_length)
	    negative_log_likelihood = negative_log_likelihood * weights.float()
	    # shape : (batch_size,)
	    per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)

	    num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
	    return per_batch_loss.sum() / num_non_empty_sequences


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_data(name):
	sentences = []
	words = set()
	max_len = 0
	with open(name,'r') as data_file:
		for l in data_file.readlines():
			l_words = list(map(lambda x: x,l.split()))
			max_len = len(l_words)+2 if len(l_words)+2 > max_len else max_len
			for w in l_words:
				words.add(w)
			sentences.extend(["@@<"] + l_words + ["/>@@"])
	word_map = {w:i+2 for i,w in enumerate(words)}
	# word_map["@@PAD"] = 0 
	word_map["@@<"] = 0
	word_map["/>@@"] = 1

	return sentences,max_len,word_map


def train_or_test(x,y,train=False):	
	index = 0
	next_state = None
	total_loss=0
	total_predict=0
	with tqdm(total=x.shape[1]//window_size) as pbar:
		while(index+window_size <= x.shape[1]):		
			cur_inp = x[:,index:index+window_size]
			cur_labels = y[:,index:index+window_size]
			if train: optimizer.zero_grad()
			loss,next_state = headNet.forward(torch.LongTensor(cur_inp),torch.LongTensor(cur_labels),h0=next_state,use_drop_out=True)
			total_loss+=loss.data*window_size*batch_size
			total_predict+=window_size*batch_size

			if train: loss.backward(),optimizer.step()
			pbar.update(1)
			index = index+window_size

			if train and index/window_size % 10 == 0:
				for i in range(1):
					generation = list(map(lambda x: invMap[x],headNet.gen()))
					print(' '.join(generation)) #filter(lambda x: '@' not in x,generation)))
					print("")

#	print(total_loss,total_predict)
	cur_perp = torch.exp(total_loss/total_predict)
	return cur_perp

		

sentences,_,word_map, = load_data('article_output.txt')

# print(sentences)

indexed_sentences = list(map(lambda x: word_map[x],sentences))
invMap = {word_map[k]:k for k in word_map.keys()}
batch_size = 11
window_size = 20
num = len(indexed_sentences)
fixed_num = num - (num % batch_size) + 1
indexed_sentences = indexed_sentences[:fixed_num]
indexed_inp = indexed_sentences[:-1]
indexed_labels = indexed_sentences[1:]

inputs = np.array(np.split(np.asarray(indexed_inp,dtype=int),batch_size))
labels = np.array(np.split(np.asarray(indexed_labels,dtype=int),batch_size))


train_inputs = inputs[:-1]
train_labels = labels[:-1]


dev_inputs = np.expand_dims(inputs[-1],0)
dev_labels = np.expand_dims(labels[-1],0)


headNet = headLM(len(word_map.keys()),64,128,window_size)



optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,headNet.parameters()), lr = 0.1)
prev_perp = 2000000
best_perp = 200000
for e in range(20):
	train_perp = train_or_test(train_inputs,train_labels,train=True)	
	#print("Train Perplexity:",perp)
	dev_perp = train_or_test(dev_inputs,dev_labels,train=False)
	best_perp = False
	if dev_perp < prev_perp:
		best_perp = True
		prev_perp = dev_perp

	print("Dev Perplexity:",dev_perp)
	save_checkpoint({
            'epoch': e,
            'state_dict': headNet.state_dict(),
            'result': prev_perp,
            'optimizer' : optimizer.state_dict(),
        }, best_perp)