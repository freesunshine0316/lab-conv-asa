
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
from asa_datastream import OID, XID
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
            if st == -1 or (st != -1 and prev_senti != cur_senti): # O Neu-I or Pos-B Neu-I or Neg-I Neu-I
                if st != -1 and prev_senti != cur_senti:
                    assert prev_senti != None
                    sentiments.append((st, i-1, prev_senti))
                prev_senti = cur_senti
                st = i
            # Neu-B Neu-I or Neu-I Neu-I
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
    n_right, n_total = 0.0, 0.0
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        batch_lens = ori_batch['input_lens']
        batch_wordlens = ori_batch['input_wordlens']
        batch_tok2word = ori_batch['input_tok2word'].numpy()
        batch_outputs = model(batch)
        loss += batch_outputs['loss'].item()
        for i, tag_ids in enumerate(batch_outputs['predictions'].cpu().tolist()): # [batch, seq]
            cur_len = batch_lens[i]
            cur_wordlen = batch_wordlens[i]

            wordtag_ids = []
            for j in range(cur_wordlen):
                x = tag_ids[batch_tok2word[i,j,0]]
                x = OID if x == XID else x
                wordtag_ids.append(x)
            prds = extract_sentiment_from_tags(wordtag_ids)
            predictions.append(prds)

            if batch['refs'] is not None:
                refs = set(tuple(x) for x in batch['refs'][i])
                refs_un = set(tuple(x[:2]) for x in batch['refs'][i])
                n_ref += len(refs)
                n_prd += len(prds)
                n_both += sum(tuple(x) in refs for x in prds)
                n_both_un += sum(tuple(x[:2]) in refs_un for x in prds)
                for j in range(cur_len):
                    ref_tag_id = ori_batch['input_tags'][i,j].item()
                    if ref_tag_id not in (OID, XID):
                        n_right += (tag_ids[j] == ref_tag_id)
                        n_total += 1.0
    model.train()
    return {'loss':loss, 'predictions':predictions, 'score':calc_f1(n_prd, n_ref, n_both),
            'score_un':calc_f1(n_prd, n_ref, n_both_un), 'accu':n_right/n_total}


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

    # load data and make_batches
    print('Loading data and making batches')
    features = asa_datastream.load_and_extract_features(args.in_path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    batches = asa_datastream.make_batch(features, FLAGS.task, FLAGS.batch_size,
            is_sort=False, is_shuffle=False)

    print("Num examples = {}".format(len(features)))
    print("Num batches = {}".format(len(batches)))

    print('Compiling model')
    if FLAGS.task == 'mention':
        model = asa_model.BertAsaMe.from_pretrained(FLAGS.pretrained_path)
    elif FLAGS.task == 'sentiment':
        model = asa_model.BertAsaSe.from_pretrained(FLAGS.pretrained_path)
    else:
        assert False, 'Unsupported task: ' + FLAGS.task
    model.load_state_dict(torch.load(args.prefix_path + ".bert_model.bin"))
    model.to(device)

    outputs = inference(model, FLAGS.model_type, batches, pro_mapping)
