
import os, sys, json
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn

import config_utils
import asa_model
import asa_datastream
from pytorch_pretrained_bert.tokenization import BertTokenizer


# Pos-B Pos-I Pos-B Neu-B Neg-I O Neg-I Pos-I Pos-B
# (0,1,+1) (2,2,+1) (3,3,0) (4,4,-1) (6,6,-1) (7,7,+1) (8,8,+1)
def extract_sentiment_from_tags(tag_ids):
    sentiments = []
    prev_senti = None
    st = -1
    for i, tid in enumerate(tag_ids):
        if asa_datastream.is_tag_begin(tid):
            cur_senti = asa_datastream.check_tag_sentiment(tid)
            if st != -1:
                assert prev_senti != None
                sentiments.append((st, i-1, prev_senti))
            prev_senti = cur_senti
            st = i
        elif asa_datastream.is_tag_inner(tid):
            cur_senti = asa_datastream.check_tag_sentiment(tid)
            if st == -1 or (st != -1 and prev_senti != cur_senti): # O Neu-I or Pos-B Neu-I
                if st != -1 and prev_senti != cur_senti:
                    assert prev_senti != None
                    sentiments.append((st, i-1, prev_senti))
                prev_senti = cur_senti
                st = i
        else:
            assert tid == 0
            if st != -1:
                assert prev_senti != None
                sentiments.append((st, i-1, prev_senti))
            prev_senti = None
            st = -1
    if st != -1:
        assert prev_senti != None
        sentiments.append((st, len(tag_ids)-1, prev_senti))
    return sentiments


# mention is predicted by boundaries, not tags
#def extract_metnion_from_tags(tag_ids):
#    mentions = []
#    st = -1
#    for i, tid in enumerate(tag_ids):
#        if asa_datastream.is_tag_begin(tid):
#            if st != -1:
#                mentions.append((st, i-1))
#            st = i
#        elif asa_datastream.is_tag_inner(tid):
#            if st == -1:
#                st = i
#        else:
#            if st != -1:
#                mentions.append((st, i-1))
#            st = -1
#    if st != -1:
#        mention.append((st, len(tag_ids)-1))
#    return mentions


def calc_f1(n_out, n_ref, n_both):
    print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both/n_out if n_out > 0.0 else 0.0
    rc = n_both/n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0*pr*rc/(pr+rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


# TODO: make consumable data from raw input
def make_data(conversation, tokenizer):
    data = {'sentences': [], # [batch, wordseq]
            'sentences_bert_idxs': [], # [batch, wordseq, wordlen]
            'sentences_bert_toks': [], # [batch, seq]
            'zp_info': []} # [a sequence of ...]
    return data


def predict_sentiment(model, batches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    loss = 0.0
    n_ref, n_prd, n_both, n_both_un = 0.0, 0.0, 0.0, 0.0
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        batch_outputs = model(batch)
        loss += batch_outputs['loss'].item()
        batch_wordseq_lengths = batch_outputs['wordseq_lengths'].cpu().tolist() # [batch]
        for i, tag_ids in enumerate(batch_outputs['predictions'].cpu().tolist()): # [batch, wordseq]
            cur_wordseq_len = batch_wordseq_lengths[i]
            prds = extract_sentiment_from_tags(tag_ids[:cur_wordseq_len])
            predictions.append(prds)
            if batch['refs'] is not None:
                refs = set(tuple(x) for x in batch['refs'][i])
                refs_un = set(tuple(x[:2]) for x in batch['refs'][i])
                n_ref += len(refs)
                n_prd += len(prds)
                n_both += sum(tuple(x) in refs for x in prds)
                n_both_un += sum(tuple(x[:2]) in refs_un for x in prds)
    model.train()
    return {'loss':loss, 'predictions':predictions, 'score':calc_f1(n_prd, n_ref, n_both),
            'score_un':calc_f1(n_prd, n_ref, n_both_un)}


def predict_mention(model, batches):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    loss = 0.0
    n_right, n_total = 0.0, 0.0
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        step_outputs = model(batch)
        loss += step_outputs['loss'].item()
        wordseq_num = step_outputs['wordseq_num']
        for i, x in enumerate(step_outputs['predictions'].cpu().tolist()):
            st = x // wordseq_num
            ed = x % wordseq_num
            predictions.append((st,ed))
            n_right += ((st,ed) in batch['refs'][i])
            n_total += 1.0
    model.train()
    print('n_right {}, n_total {}'.format(n_right, n_total))
    return {'loss':loss, 'predictions':predictions, 'score':n_right/n_total}


#is_test = False
#if is_test:
#    from asa_datastream import TAG_MAPPING
#    tag_seq = ['PosB', 'PosI', 'PosB', 'NeuB', 'NegI', 'O', 'NegI', 'PosI', 'PosB']
#    tag_ids = [TAG_MAPPING[x] for x in tag_seq]
#    sentiments = extract_sentiment_from_tags(tag_ids)
#    print(sentiments)
#    print('(0,1,+1) (2,2,+1) (3,3,0) (4,4,-1) (6,6,-1) (7,7,+1) (8,8,+1)')
#    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, required=True, help='Prefix path to the saved model')
    parser.add_argument('--in_path', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--out_path', type=str, default=None, help='Path to the output file.')
    args, unparsed = parser.parse_known_args()
    FLAGS = config_utils.load_config(args.prefix_path + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if 'bert' in FLAGS.pretrained_path:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))

    # ZP setting
    is_only_azp = False

    # load data and make_batches
    print('Loading data and making batches')
    data_type = 'resolution_inference'
    data = make_data(args.in_path, tokenizer)
    features = zp_datastream.extract_features(data, tokenizer,
            char2word=FLAGS.char2word, data_type=data_type, is_only_azp=is_only_azp)
    batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
            is_sort=False, is_shuffle=False)

    print('Compiling model')
    model = zp_model.BertZP.from_pretrained(FLAGS.pretrained_path, char2word=FLAGS.char2word,
            pro_num=len(pro_mapping), max_relative_position=FLAGS.max_relative_position)
    model.load_state_dict(torch.load(args.prefix_path + ".bert_model.bin"))
    model.to(device)

    outputs = inference(model, FLAGS.model_type, batches, pro_mapping)
