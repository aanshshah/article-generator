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
		self.decoder = nn.GRU(word_embed_sz,hidden_size,num_layers=1,batch_first=True)
		self.hidden2vocab = nn.Linear(hidden_size,vocab_sz)
		self.dropout = nn.Dropout(0.25) 


	def forward(self,batched_inp,batched_labels,padding=0,h0=None,use_drop_out=False):	
		batch_size = batched_inp.size(0)
		embedded = self.embed_words(batched_inp)
		if use_drop_out:
			embedded = self.dropout(embedded)
		out_states,_ = self.decoder(embedded)
		logits = self.hidden2vocab(out_states)
		mask = (batched_labels != padding).double()
		# print(mask)
		# print(batched_inp)
		# print(batched_labels)
		loss = self.sequence_cross_entropy_with_logits_or_prob(logits, batched_labels, mask)
		return loss

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

	def sequence_cross_entropy_with_logits_or_prob(self,logits,targets,weights,batch_average=False,label_smoothing=None,takeLogits=True):
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

	    if batch_average:
	        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
	        return per_batch_loss.sum() / num_non_empty_sequences
	    return per_batch_loss.sum()


def load_data(name):
	sentences = []
	words = set()
	authors = []
	max_len = 0
	with open(name,'r') as data_file:
		for l in data_file.readlines():
			if ":" in l:
				authors.append(l[:l.index(":")+1])
				l = l[l.index(":")+1:]

			l_words = list(map(lambda x: x.lower(),l.split()))
			max_len = len(l_words)+4 if len(l_words)+4 > max_len else max_len
			for w in l_words:
				words.add(w)
			sentences.append(["@@PAD","@<"] + l_words + ["/>@","@@PAD"])
	word_map = {w:i+3 for i,w in enumerate(words)}
	word_map["@@PAD"] = 0 
	word_map["@<"] = 1
	word_map["/>@"] = 2

	return sentences,max_len,word_map,authors

def pad_process_sentences(sentences,max_len,word_map):
	proc_sentences = []
	for s in sentences:
		new_sen = np.zeros(max_len)
		new_sen[:len(s)] += np.asarray(list(map(lambda x: word_map[x],s)),dtype=int)
		proc_sentences.append(new_sen)
	return np.array(proc_sentences,dtype=int)


sentences,max_len,word_map,authors = load_data('output_new.txt')

proc_sentences = pad_process_sentences(sentences,max_len,word_map)

invMap = {word_map[k]:k for k in word_map.keys()}

batch_size = 20
num = proc_sentences.shape[0]
fixed_num = num - (num % batch_size)
print(num,fixed_num)

proc_sentences = proc_sentences[:num - (num % batch_size)]
inputs = proc_sentences[:,:-1].reshape(-1,batch_size,max_len-1)
labels = proc_sentences[:,1:].reshape(-1,batch_size,max_len-1)

headNet = headLM(len(word_map.keys()),32,64,max_len)



optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,headNet.parameters()), lr = 0.01)
train_data = np.arange(inputs.shape[0]-50)
dev_data = np.arange(inputs.shape[0]-50,inputs.shape[0])
prev_perp = 10
for e in range(3):
	total_predict = 0
	total_loss = 0
	with tqdm(total=train_data.shape[0]) as pbar:
		np.random.shuffle(train_data)
		for batch_num in train_data:		
			optimizer.zero_grad()
			loss = headNet.forward(torch.LongTensor(inputs[batch_num]),torch.LongTensor(labels[batch_num]),use_drop_out=True)
			loss.backward()
			optimizer.step()
			pbar.update(1)
		
		for batch_num in dev_data:
			loss = headNet.forward(torch.LongTensor(inputs[batch_num]),torch.LongTensor(labels[batch_num]))
			total_loss+=loss.data
			total_predict+=np.count_nonzero(inputs[batch_num])
	cur_perp = np.exp(total_loss/total_predict)

	print(cur_perp )
	if cur_perp > prev_perp:
		break
	prev_perp = cur_perp

	for i in range(50):
		generation = list(map(lambda x: invMap[x],headNet.gen()))
		print(random.choice(authors),' '.join(filter(lambda x: '@' not in x,generation)))
	print("")
	
