from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.evaluator import eval_wc

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Co-training LM+LSTM-CRF+K-mer match')
    parser.add_argument('--rand_embedding', default=False, help='random initialize word embedding')
    parser.add_argument('--emb_file', default='../Embedding/building_wordvec_30d_apm.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', default='../Data/apm-ebu3b-tag/ap_m.train', help='path to training file')
    parser.add_argument('--dev_file', default='../Data/apm-ebu3b-tag/ap_m.dev', help='path to development file')
    parser.add_argument('--test_file', default='../Data/apm-ebu3b-tag/ebu3b.test', help='path to test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=150, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--drop_out', type=float, default=0.55, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=30, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', default=False, help='fine tune the diction of word embedding or not')
    parser.add_argument('--load_check_point', default=False, help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', default=True, help='also load optimizer from the checkpoint')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--lambda0', type=float, default=0.9, help='lambda0')
    parser.add_argument('--co_train', default=True, help='cotrain language model')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--high_way', default=True, help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    parser.add_argument('--cotrain_file_1',default='../Data/apm-ebu3b-tag/apm.test',help='cotrain file1 path')
    parser.add_argument('--word_emb_weight',default=1, type=float, help='weight of word embedding to LM output')
    parser.add_argument('--lambda0_min_value',type=float, default=0.8,help='lambda0 lower bound')
    parser.add_argument('--lambda0_dr',type=float, default=0.01,help='lambda0 descending rate')
    parser.add_argument('--no_dict', action='store_true', help='do not use prior knowledge')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    
    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        train_lines = f.readlines()
    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()
    with codecs.open(args.cotrain_file_1, 'r', 'utf-8') as f:
        cotrain_lines=f.readlines()
    
    #load prior knowledge
    with open('../Data/source_full.json') as f:
        knowledge_dict = json.load(f)

    train_features, train_labels=utils.read_corpus(train_lines)
    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)
    co_features,co_labels=utils.read_corpus(cotrain_lines+dev_lines+test_lines)
    
    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            l_map = checkpoint_file['l_map']
            c_map = checkpoint_file['c_map']
            in_doc_words = checkpoint_file['in_doc_words']
            train_features, train_labels = utils.read_corpus(train_lines)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
    else:
        print('constructing coding table')

        # converting format
        all_features, all_labels, f_map, l_map, c_map = utils.generate_corpus_char(train_lines+dev_lines+test_lines+cotrain_lines, if_shrink_c_feature=True, c_thresholds=args.mini_count, if_shrink_w_feature=False)
        with open('c_map.json','w') as f:
            json.dump(c_map,f)
        f_set = {v for v in f_map}
        f_map = utils.shrink_features(f_map, all_features, args.mini_count)
        if args.rand_embedding:
            print("embedding size: '{}'".format(len(f_map)))
            in_doc_words = len(f_map)
        else:
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
            dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
            print("feature size: '{}'".format(len(f_map)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                f_map = {'<eof>': 0}
            
            f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', f_map, dt_f_set, args.caseless, args.unk, args.word_dim, shrink_to_corpus=args.shrink_embedding)
            print("embedding size: '{}'".format(len(f_map)))
            


        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)
        for label in l_set:
            if label not in l_map:
                l_map[label] = len(l_map)
    
    print('constructing dataset')
    # construct dataset
    dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(train_features, train_labels, l_map, c_map, f_map, args.caseless)
    dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(dev_features, dev_labels, l_map, c_map, f_map, args.caseless)
    test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(test_features, test_labels, l_map, c_map, f_map, args.caseless)
    co_dataset,_,_=utils.construct_bucket_mean_vb_wc(co_features,co_labels,l_map,c_map,f_map,args.caseless)
    
    dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, 1, shuffle=False, drop_last=False) for tup in test_dataset]
    co_dataset_loader=[torch.utils.data.DataLoader(tup,args.batch_size,shuffle=True,drop_last=False) for tup in co_dataset]
    
    
    #string match part
    word_to_id,emb_tensor,emb =utils.string_match_embedding(args.emb_file,args.word_dim)
    
    # build model
    print('building model')
   
    ner_model = LM_LSTM_CRF(args.word_emb_weight,len(l_map), len(c_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim, args.word_hidden, args.word_layers, emb_tensor.shape[0], args.drop_out, large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words, highway_layers = args.highway_layers)
    
    if args.load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_word_embedding(emb_tensor)
        ner_model.rand_init(init_word_embedding=args.rand_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])

    crit_lm = nn.CrossEntropyLoss()
    crit_ner = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        
        crit_ner.cuda()
        crit_lm.cuda()
        ner_model.cuda()
        packer = CRFRepack_WC(len(l_map), True)

    else:
        if_cuda = False
        packer = CRFRepack_WC(len(l_map), False)

    tot_length = sum(map(lambda t: len(t), dataset_loader))

    best_f1 = float('-inf')
    best_acc = float('-inf')
    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    evaluator = eval_wc(packer, c_map, l_map)
    evaluator_filter = eval_wc(packer, c_map, l_map, ['B-site', 'I-site', 'O'])
    print("start training...")
    loss_list=[]
    crf_loss_list=[]
    lm_loss_list=[]

    f1_test_list=[]
    r_c_map = utils.revlut(c_map)
    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        crf_loss=0
        lm_loss=0
        ner_model.train()
        args.lambda0=max(1-args.lambda0_dr*args.start_epoch,args.lambda0_min_value)
        data={"train":dataset_loader,"co":co_dataset_loader}
        for i in data:
          
          for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in tqdm(
                itertools.chain.from_iterable(data[i]), mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):
            mask_v=mask_v.bool()
            f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)
            ner_model.zero_grad()
        
            w_f_word=utils.reconstruct_word_input(w_f,f_map,emb,word_to_id,args.gpu)
        
            
            prior_prob = utils.generate_prior_prob(r_c_map, l_map, f_f, knowledge_dict) 
        
            scores = ner_model(f_f, f_p, b_f, b_p, w_f,w_f_word, prior_prob)
        
            loss =(1-args.lambda0)*crit_ner(scores, tg_v, mask_v, prior_prob)
        
        
            if args.co_train:
                cf_p = f_p[0:-1, :].contiguous()
                cb_p = b_p[1:, :].contiguous()
                cf_y = w_f[1:, :].contiguous()
                cb_y = w_f[0:-1, :].contiguous()

            
            

                cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                cbs, _ = ner_model.word_pre_train_backward(b_f,cb_p)
                
                cfs_loss=args.lambda0 * crit_lm(cfs, cf_y.view(-1))
                cbs_loss=args.lambda0 * crit_lm(cbs, cb_y.view(-1))
                lm_loss+=utils.to_scalar(cfs_loss)+utils.to_scalar(cbs_loss)
                if i=='train':
                    crf_loss+=utils.to_scalar(loss)
                    loss = loss + cfs_loss
                else:
                    loss=cfs_loss
                loss = loss + cbs_loss
                
            epoch_loss+=utils.to_scalar(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(ner_model.parameters(), args.clip_grad)
            optimizer.step()

        

            


        epoch_loss /= tot_length
        crf_loss/=tot_length
        lm_loss/=tot_length
        loss_list.append((args.start_epoch,round(epoch_loss,2)))
        crf_loss_list.append((args.start_epoch,round(crf_loss,2)))
        lm_loss_list.append((args.start_epoch,round(lm_loss,2)))
    
        
        # update lr
        if args.update == 'sgd':
            utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # eval & save check_point

        dev_result = evaluator.calc_score(ner_model, dev_dataset_loader,False,f_map,emb,word_to_id,args.gpu, r_c_map, l_map, knowledge_dict, args.no_dict)
        # for label, (dev_f1, dev_pre, dev_rec, dev_acc, msg) in dev_result.items():
        #     print('DEV : %s : dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f | %s\n' % (label, dev_f1, dev_rec, dev_pre, dev_acc, msg))
        (dev_f1, dev_pre, dev_rec, dev_acc, msg) = dev_result['total']

        if dev_f1 > best_f1:
            patience_count = 0
            best_f1 = dev_f1

            test_result = evaluator.calc_score(ner_model, test_dataset_loader,True,f_map,emb,word_to_id,args.gpu, r_c_map, l_map, knowledge_dict, args.no_dict)
            test_result_filted = evaluator_filter.calc_score(ner_model, test_dataset_loader,True,f_map,emb,word_to_id,args.gpu, r_c_map, l_map, knowledge_dict, args.no_dict)
            # for label, (test_f1, test_pre, test_rec, test_acc, msg) in test_result.items():
            #     print('TEST : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' % (label, test_f1, test_rec, test_pre, test_acc, msg))
            (test_f1, test_rec, test_pre, test_acc, msg) = test_result['total']
            (test_f1_filted, _, _, test_acc_filted, _) = test_result_filted['total']

            track_list.append(
                {'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                    'test_acc': test_acc})

            print(
                '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f), saving...' %
                (epoch_loss,
                    args.start_epoch,
                    dev_f1,
                    dev_acc,
                    test_f1,
                    test_acc))
            ## print filted score
            print(
                '(Filtered Score: F1 on test = %.4f, acc on test= %.4f)' %
                (test_f1_filted,
                    test_acc_filted))

            f1_test_list.append((args.start_epoch,round(test_f1,4)))
            print("total_test:"+str(f1_test_list))
            print("total_loss:"+str(loss_list))
            print("crf_loss:"+str(crf_loss_list))
            print("lm_loss:"+str(lm_loss_list))

            # try:
            
            utils.save_checkpoint({
                'epoch': args.start_epoch,
                'state_dict': ner_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'f_map': f_map,
                'l_map': l_map,
                'c_map': c_map,
                'in_doc_words': in_doc_words,
                'word_num': emb_tensor.shape[0],
            }, {'track_list': track_list,
                'args': vars(args)
                }, args.checkpoint + 'cwlm_lstm_crf')
            # except Exception as inst:
            #     print(inst)
            
        else:
            patience_count += 1
            print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f)' %
                    (epoch_loss,
                    args.start_epoch,
                    dev_f1,
                    dev_acc))
            track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break

    #print best
    eprint(args.checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))
    # printing summary
    print('setting:')
    print(args)
