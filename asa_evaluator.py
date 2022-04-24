
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
from asa_datastream import TAGS
from pytorch_pretrained_bert.tokenization import BertTokenizer


# Pos-B Pos-I Pos-B Neu-B Neg-I O Neg-I Pos-I Pos-B
# (0,1,+1) (2,2,+1) (3,3,0) (4,4,-1) (6,6,-1) (7,7,+1) (8,8,+1)
def extract_sentiment_from_tags(tag_ids):
    sentiments = []
    prev_senti = None
    st = -1
    for i, tid in enumerate(tag_ids):
        if asa_datastream.is_tag_begin(tid):
            if st != -1:
                assert prev_senti != None
                sentiments.append((st, i-1, prev_senti))
            prev_senti = asa_datastream.check_tag_sentiment(tid)
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
    #print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both/n_out if n_out > 0.0 else 0.0
    rc = n_both/n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0*pr*rc/(pr+rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


def predict_sentiment(model, tokenizer, batches, verbose=0, senti=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    loss = 0.0
    n_ref, n_prd, n_both, n_both_un = 0.0, 0.0, 0.0, 0.0
    n_right, n_total = 0.0, 0.0
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        batch_outputs = model(batch)
        loss += batch_outputs['loss'].item()
        batch_seq_lengths = batch_outputs['seq_lengths'].cpu().tolist() # [batch]
        for i, tag_ids in enumerate(batch_outputs['predictions'].cpu().tolist()): # [batch, seq]
            seq_len = batch_seq_lengths[i]
            prds = extract_sentiment_from_tags(tag_ids[:seq_len])
            if senti != None:
                prds = set([x for x in prds if x[2] == senti])
            predictions.append(prds)
            if batch['refs'] is not None:
                refs = set(tuple(x) for x in batch['refs'][i])
                if senti != None:
                    refs = set([x for x in refs if x[2] == senti])
                refs_un = set(tuple(x[:2]) for x in refs)
                n_ref += len(refs)
                n_prd += len(prds)
                n_both += sum(tuple(x) in refs for x in prds)
                n_both_un += sum(tuple(x[:2]) in refs_un for x in prds)
                for j in range(seq_len):
                    ref_tag = ori_batch['input_tags'][i,j].item()
                    if ref_tag != 0:
                        n_right += (tag_ids[j] == ref_tag)
                        n_total += 1.0
            if verbose and set(prds) != refs:
                assert batch['refs'] is not None
                print(tokenizer.decode(batch['input_ids'][i]))
                print(refs)
                print(prds)
                print(' '.join('{}({})'.format(TAGS[x],j) for j, x in enumerate(tag_ids[:seq_len])))
                print('===========')
    model.train()
    if batches[0]['refs'] is not None:
        f1 = calc_f1(n_prd, n_ref, n_both)
        f1_un = calc_f1(n_prd, n_ref, n_both_un)
        print('F1 {}, n_prd {}, n_ref {}, n_both {}; F1-un {}'.format(f1, n_prd, n_ref, n_both, f1_un))
        return {'loss':loss, 'predictions':predictions, 'score':f1, 'score_un':f1_un, 'accu':n_right/n_total}
    else:
        return {'predictions':predictions}


def predict_mention(model, tokenizer, batches, verbose=0, senti=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    loss = 0.0
    n_right, n_total = 0.0, 0.0
    n_right_same, n_total_same = 0.0, 0.0
    n_right_cross, n_total_cross = 0.0, 0.0
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        step_outputs = model(batch)
        loss += step_outputs['loss'].item()
        seq_num = step_outputs['seq_num']
        for i, x in enumerate(step_outputs['predictions'].cpu().tolist()):
            if senti != None and senti != batch['senti'][i]:
                continue
            st = x // seq_num
            ed = x % seq_num
            pred = tokenizer.decode(batch['input_ids'][i,st:ed+1])
            predictions.append(pred)
            is_correct = pred in batch['refs'][i]
            #print(pred)
            #print(batch['refs'][i])
            n_right += is_correct
            n_total += 1.0
            n_right_same += is_correct & (batch['is_cross'][i] == False)
            n_total_same += (batch['is_cross'][i] == False)
            n_right_cross += is_correct & batch['is_cross'][i]
            n_total_cross += batch['is_cross'][i]
            if verbose and not is_correct:
                print(batch['senti_lex'][i])
                print(batch['refs'][i])
                print(pred)
                print('========')
    model.train()
    accu = n_right/n_total
    accu_same = n_right_same/n_total_same
    accu_cross = n_right_cross/n_total_cross
    print('Accuracy {}, n_right {}, n_total {}'.format(accu, n_right, n_total))
    print('Accuracy-same {}, n_right_same {}, n_total_same {}'.format(accu_same, n_right_same, n_total_same))
    print('Accuracy-cross {}, n_right_cross {}, n_total_cross {}'.format(accu_cross, n_right_cross, n_total_cross))
    return {'loss':loss, 'predictions':predictions, 'score':accu}


def enrich_flag(FLAGS):
    if hasattr(FLAGS, 'freeze_bert') == False:
        FLAGS.freeze_bert = False

    if hasattr(FLAGS, 'use_embedding') == False:
        FLAGS.use_embedding = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, required=True, help='Prefix path to the saved model')
    parser.add_argument('--in_path', type=str, default=None, help='Path to the input file')
    parser.add_argument('--senti_of_interest', type=int, default=None, help='Sentiment of interest for calculating scores')
    args, unparsed = parser.parse_known_args()
    FLAGS = config_utils.load_config(args.prefix_path + ".config.json")
    enrich_flag(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if 'bert' in FLAGS.pretrained_path:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    # load data and make_batches
    print('Loading data and making batches')
    in_path = args.in_path if args.in_path is not None else FLAGS.test_path
    features = asa_datastream.load_and_extract_features(in_path, tokenizer,
            FLAGS.task)
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

    if FLAGS.freeze_bert:
        model.freeze_bert()
    elif FLAGS.use_embedding:
        model.setup_embedding(len(tokenizer.vocab))

    model.load_state_dict(torch.load(args.prefix_path + ".bert_model.bin"))#, map_location='cpu'))
    model.to(device)

    if FLAGS.task == 'mention':
        predict_mention(model, batches, verbose=0, senti=args.senti_of_interest)
    else:
        predict_sentiment(model, batches, verbose=0, senti=args.senti_of_interest)

