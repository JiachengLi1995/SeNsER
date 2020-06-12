"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)

.. moduleauthor:: Liyuan Liu, Frank Xu
"""


import torch
import numpy as np
import itertools

import model.utils as utils
from torch.autograd import Variable

from model.crf import CRFDecode_vb

# np.random.seed(999)
# torch.manual_seed(999)
# torch.cuda.manual_seed_all(999)
class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
    """


    def __init__(self, packer,c_map, l_map, removed_label):

        self.removed_label = removed_label
        self.packer = packer
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)
        self.c_map = c_map
        self.r_c_map = utils.revlut(c_map)
        self.totalp_counts={}
        self.truep_counts={}
        self.fn_counts={}
        self.fp_counts={}
        self.f1={}
        

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0
        self.totalp_counts={}
        self.truep_counts={}
        self.fn_counts={}
        self.fp_counts={}
        self.f1={}

    def calc_f1_batch(self, decoded_data, target_data):
        """
        update statics for f1 score

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)
        
        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
        
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length]
            best_path = decoded[:length]

            ## filter out removed label
            best_path_filted, gold_filted = self.label_filter(best_path.numpy(), gold.numpy())
            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path_filted, gold_filted)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def label_filter(self, best_path, gold):
        best_path_filted = []
        gold_filted = []
        if len(self.removed_label)==0:
            return best_path, gold
        for i in range(len(best_path)):
            gold_label = self.r_l_map[gold[i]]
            #print(gold_label)
            if gold_label not in self.removed_label:
                best_path_filted.append(best_path[i])
                gold_filted.append(gold[i])
        
        return np.array(best_path_filted), np.array(gold_filted)
        
    def calc_acc_batch(self, decoded_data, target_data):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = self.packer.convert_for_eval(target)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        if self.guess_count == 0:
            return {'total': (0.0, 0.0, 0.0, 0.0, '')}
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return {'total': (0.0, 0.0, 0.0, 0.0, '')}
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        message=""
        self.f1['total'] = (f, precision, recall, accuracy, message)
        for label in self.totalp_counts:
            tp = self.truep_counts.get(label,1)
            fn = sum(self.fn_counts.get(label,{}).values())
            fp = sum(self.fp_counts.get(label,{}).values())
            # print(label, str(tp), str(fp), str(fn), str(self.totalp_counts.get(label,0)))
            precision = tp / float(tp+fp+1e-9)
            recall = tp / float(tp+fn+1e-9)
            f = 2 * (precision * recall) / (precision + recall+1e-9)
            message = str(self.fn_counts.get(label, {}))
            self.f1[label] = (f, precision, recall, 0, message)
        return self.f1

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy

    def eval_instance(self, best_path, gold):
        """
        update statics for one instance

        args:
            best_path (seq_len): predicted
            gold (seq_len): ground-truth
        """
        
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))

        for i in range(total_labels):
            gold_label = self.r_l_map[gold[i]]
            guessed_label = self.r_l_map[best_path[i]]
            self.totalp_counts[gold_label] = 1 + self.totalp_counts.get(gold_label,0)
            if gold_label == guessed_label:
                self.truep_counts[gold_label] = 1 + self.truep_counts.get(gold_label,0)
            else:
                val = self.fn_counts.get(gold_label,{})
                val[guessed_label] = 1+ val.get(guessed_label,0)
                self.fn_counts[gold_label]=val

                val2 = self.fp_counts.get(guessed_label,{})
                val2[gold_label] = 1+ val2.get(gold_label,0)
                self.fp_counts[guessed_label] = val2

        gold_chunks = utils.iobes_to_spans(gold, self.r_l_map)
        # gold_chunks=set([i for i in gold_chunks if i[0:i.find('@')]!='other'])
        gold_count = len(gold_chunks)

        guess_chunks = utils.iobes_to_spans(best_path, self.r_l_map)
        # guess_chunks=set([i for i in guess_chunks if i[0:i.find('@')]!='other'])
        guess_count = len(guess_chunks)
        
        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)
        
        return correct_labels, total_labels, gold_count, guess_count, overlap_count

class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        for feature, tg, mask in itertools.chain.from_iterable(dataset_loader):
            fea_v, _, mask_v = self.packer.repack_vb(feature, tg, mask)
            scores, _ = ner_model(fea_v)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()

class eval_wc(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, c_map, l_map, removed_label = []):
        eval_batch.__init__(self, packer, c_map, l_map, removed_label)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        self.eval_b = self.calc_f1_batch
        self.calc_s = self.f1_score
        

    def calc_score(self, ner_model, dataset_loader,out,f_map,emb,word_to_id,gpu, r_c_map, l_map, knowledge_dict, no_dict=False):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()
        
        
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v in itertools.chain.from_iterable(dataset_loader):
            mask_v=mask_v.bool()  
            f_f, f_p, b_f, b_p, w_f, _, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v)
            w_f_word=utils.reconstruct_word_input(w_f,f_map,emb,word_to_id,gpu)  
            prior_prob = utils.generate_prior_prob(r_c_map, l_map, f_f, knowledge_dict)            
            scores = ner_model(f_f, f_p, b_f, b_p, w_f,w_f_word, prior_prob)
            decoded = self.decoder.decode(scores.data, mask_v.data, prior_prob, no_dict)
            self.eval_b(decoded, tg)
        
        return self.calc_s()


    def check_output(self, ner_model, dataset_loader,out,f_map,emb,word_to_id,gpu, knowledge_dict, no_dict=False):
        ner_model.eval()
        self.reset()
        f = open('model_output.txt','w')
        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v in itertools.chain.from_iterable(dataset_loader):
            mask_v=mask_v.bool()
            
            f_f, f_p, b_f, b_p, w_f, _, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v)
            w_f_word=utils.reconstruct_word_input(w_f,f_map,emb,word_to_id,gpu)       
            prior_prob = utils.generate_prior_prob(self.r_c_map, self.l_map, f_f, knowledge_dict)     
            scores = ner_model(f_f, f_p, b_f, b_p, w_f,w_f_word, prior_prob)
            decoded = self.decoder.decode(scores.data, mask_v.data, prior_prob, no_dict)
           
            self.eval_b(decoded, tg)
            batch_decoded = torch.unbind(decoded, 1)
            batch_targets = torch.unbind(tg, 0)
            batch_f_f = torch.unbind(f_f, 1)
            
            for decoded, target, character in zip(batch_decoded, batch_targets, batch_f_f):
                
                gold = self.packer.convert_for_eval(target)
                # remove padding
                length = utils.find_length_from_labels(gold, self.l_map)
                gold = gold[:length].numpy()
                best_path = decoded[:length].numpy()
                character_filted = []
                for c in character.cpu().numpy():
                    if c!=42:
                        character_filted.append(c)
                char = character_filted[:length]
                for i in range(len(gold)):
                    
                    f.write(self.r_c_map[char[i]]+' '+self.r_l_map[gold[i]]+' '+self.r_l_map[best_path[i]])
                    f.write('\n')
        
                f.write('\n')
        f.close()
        return self.calc_s()