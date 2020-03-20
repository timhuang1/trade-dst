import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
import pprint

class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(TRADE, self).__init__()
        '''
        Instantiation in myTrain.py:
            model = globals()[args['decoder']](
                hidden_size=int(args['hidden']), 
                lang=lang, 
                path=args['path'], 
                task=args['task'], 
                lr=float(args['learn']), 
                dropout=float(args['drop']),
                slots=SLOTS_LIST,
                gating_dict=gating_dict, 
                nb_train_vocab=max_word)
        where:
            hidden_size # Using hidden size = 400 for pretrained word embedding (300 + 100)...
            # lang, mem_lang = Lang(), Lang()
            lang.index_words(ALL_SLOTS, 'slot')
            mem_lang.index_words(ALL_SLOTS, 'slot')
            path = # parser.add_argument('-path', help='path of the file to load')
            lr = # -lr=0.001
            dropout =   # 'drop': 0.2
            SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
            gating_dict = {"ptr":0, "dontcare":1, "none":2}
            nb_train_vocab = lang.n_words  # Vocab_size Training 15462
        '''
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size    
        self.lang = lang[0]
        self.mem_lang = lang[1] 
        self.lr = lr 
        self.dropout = dropout
        self.slots = slots[0]  # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate) 
        
        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path)+'/enc.th')
                trained_decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                trained_decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
            
            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        
        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg,print_loss_ptr,print_loss_gate)
    
    def save_model(self, dec_type):
        directory = 'save/TRADE-'+args["addName"]+args['dataset']+str(self.task)+'/'+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')
    
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        '''
        call in myTrain.py:
            model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))
        where:
            for i, data in enumerate(train) # train is torch.utils.data.DataLoader derived from train samples
            int(args['clip']) = 'clip': 10
            SLOTS_LIST[1] # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
            reset=(i==0)  # for i, data in enumerate(train)
        '''
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]  # 'teacher_forcing_ratio': 0.5
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing, slot_temp)

        # 
        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(), #[:,:len(self.point_slots)].contiguous(),
            data["y_lengths"]) #[:,:len(self.point_slots)])

        # print ("data[generate_y]: {}".format(data["generate_y"])) 
        # print ("data[y_lengths]: {}".format(data["y_lengths"]))
        # print ("all_point_outputs: {}".format(all_point_outputs))
        # print ("loss_ptr: {}".format(loss_ptr))

        # # data[generate_y].shape: torch.Size([32, 30, 6])  # i.e., [batch_size, slot_num, max_len of value]
        # # data[y_lengths].shape: torch.Size([32, 30]) [batch_size, slot_num]
        # # all_point_outputs.shape: torch.Size([30, 32, 6, 18311]) [slot_num, batch_size, max_len of value, vocab_num]
        # # loss_ptr: 10.12643051147461

        '''
        # PAD_token = 1
        # EOS_token = 2
        data[generate_y]: tensor([[[ 212,    2,    1,    1,    1,    1], 
         [ 212,    2,    1,    1,    1,    1],
         [ 212,    2,    1,    1,    1,    1],
         ...,
         [ 212,    2,    1,    1,    1,    1],
         [ 212,    2,    1,    1,    1,    1],
         [ 212,    2,    1,    1,    1,    1]], ...)
    
        # reason for the least number is two: every value ends with EOS_token
        data[y_lengths]: tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2,
         6, 3, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2,
         3, 6, 2, 2, 2, 2], ...) 
        '''

        # three-class classification: {"ptr":0, "dontcare":1, "none":2}
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:  # 'use_gate': 1
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr
        
        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()
    
    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def optimize_GEM(self, clip):
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        '''
        call in self.train_batch():
            all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing, slot_temp)
        where:
            for i, data in enumerate(train) # train is torch.utils.data.DataLoader derived from train samples
            use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]  # 'teacher_forcing_ratio': 0.5 
            slot_temp = SLOTS_LIST[1] # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
        '''
        if args['unk_mask'] and self.decoder.training: # 'unk_mask': 1; to apply word-level dropout (ratio: self.dropout=  # 'drop': 0.2)
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)

            if USE_CUDA: 
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
            encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'], \
            use_teacher_forcing, slot_temp) 

        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        '''
        call in myTest.py:
            acc_test = model.evaluate(test, 1e7, SLOTS_LIST[3]) 
        where:
            # test is torch.utils.data.DataLoader derived from test samples
            SLOTS_LIST[3]  # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]

        '''
        self.encoder.train(False)
        self.decoder.train(False)  
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        # gating_dict = {"ptr":0, "dontcare":1, "none":2}
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)
            print ("len(words): {}".format([len(words[i]) for i in range(3)] ))
            # type(words): list
            # len(words): 30
            # [len(words[i]) for i in range(3)]: [10, 10, 10]
            # words shape: [30, 10, 32] i.e., [slot-num, max-length, batch-size]

            for bi in range(batch_size):  # bi stands for batch index
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                print ("dial ID: {}".format(data_dev["ID"][bi]))
                print ("Turn ID: {}".format(data_dev["turn_id"][bi]))
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                print ("turn_belief: {}".format(data_dev["turn_belief"][bi]))
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)
                print ("gates.shape: {}".format(gates.shape))
                # gates.shape: torch.Size([30, 32, 3])


                # pointer-generator results
                if args["use_gate"]:  # 'use_gate': 1
                    for si, sg in enumerate(gate): # si, sg stands for slot-index, slot-gate
                        if sg==self.gating_dict["none"]:
                            continue
                        elif sg==self.gating_dict["ptr"]: 
                            pred = np.transpose(words[si])[bi]
                            st = []    # st stands for slot-text
                            for e in pred:
                                if e== 'EOS': break
                                else: st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()]) # ''-''-'dontcare'
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                # print ("predict_belief_bsz_ptr: {}\n".format(predict_belief_bsz_ptr))

                '''
                dial ID: PMUL2437.json
                Turn ID: 10
                turn_belief: ['restaurant-pricerange-moderate', 'restaurant-area-centre', 'attraction-type-architecture', 'attraction-name-all saints church', 'attraction-area-centre']
                predict_belief_bsz_ptr: ['attraction-area-centre', 'restaurant-pricerange-moderate', 'restaurant-area-centre', 'attraction-name-all', 'attraction-type-architecture']
                '''

                # print ("all_prediction: {}".format(all_prediction))
                '''
                all_prediction: {'PMUL2437.json': {10: {'turn_belief': ['restaurant-pricerange-moderate', 'restaurant-area-centre', 'attraction-type-architecture', 'attraction-name-all saints church', 'attraction-area-centre'], 
                'pred_bs_ptr': ['attraction-area-centre', 'restaurant-pricerange-moderate', 'restaurant-area-centre', 'attraction-name-all', 'attraction-type-architecture']}}}
                '''

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]: # 'genSample': 0
                    print("True", set(data_dev["turn_belief"][bi]) )
                    print("Pred", set(predict_belief_bsz_ptr), "\n")  


            raise ValueError("Intended pause in myTrain.py!")

        if args["genSample"]: # 'genSample': 0
            json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score


    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        '''
        call in self.evaluate():
            joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)
        where:

            slot_temp = SLOTS_LIST[3]  # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
        '''
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()      
        '''
        Instantiation in :
            self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        where:
            lang.n_words  # Vocab_size Training 15462
            hidden_size # Using hidden size = 400 for pretrained word embedding (300 + 100)...
            dropout 0.2
        '''
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:  # 'load_embedding': 1
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True

        if args["fix_embedding"]: # 'fix_embedding': 0
            self.embedding.weight.requires_grad = False

        print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)   
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs.transpose(0,1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        '''
        Instantiation in :
            self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate) 
        where:
            # lang, mem_lang = Lang(), Lang()
            lang.index_words(ALL_SLOTS, 'slot')
            mem_lang.index_words(ALL_SLOTS, 'slot') 
            self.encoder.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
            self.lang.n_words
            hidden_size # 400
            dropout # 0.2
            self.slots = slots[0]  # SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
            self.nb_gate = len(gating_dict) # gating_dict = {"ptr":0, "dontcare":1, "none":2}
        '''
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slot_temp):
        '''
        call in self.encode_and_decode():
            all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
                                encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'], \
                                use_teacher_forcing, slot_temp)
        where:


        '''
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA: 
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()
        
        # Get the slot embedding 
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA: domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        if args["parallel_decode"]:
            # Compute pointer-generator output, puting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size) # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1) # 1 * (batch*|slot|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []
            
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0: 
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                
                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]
                
                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])
                
                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, (len(slot_temp), batch_size, self.vocab_size))
                
                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1,0)))
                else:
                    decoder_input = self.embedding(pred_word)   
                
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0: 
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size())
                    if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                    p_context_ptr.scatter_add_(1, story, prob)  # word prob from dialogue history 
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)   
                    if USE_CUDA: decoder_input = decoder_input.cuda()
                counter += 1
                words_point_out.append(words)
        
        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
