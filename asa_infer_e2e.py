
import os, sys, json
import argparse
import numpy as np
import time
import random
import string
from collections import defaultdict

import torch
import torch.nn as nn

import config_utils
import asa_model
import asa_datastream
import asa_evaluator

from asa_datastream import TAGS
from pytorch_pretrained_bert.tokenization import BertTokenizer


def decode_dialogue(args, dialogue, sentiment_model, mention_model, tokenizer):
    print('Dialogue length : {}'.format(len(dialogue['conv'])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLS_ID, SEP_ID = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    # decode sentiment
    features = []
    for i, turn in enumerate(dialogue['conv']):
        cur_ids, cur_tok2word = asa_datastream.bert_tokenize(turn, tokenizer, args.tok2word_strategy) # [tok_seq], [word_seq, word_len]
        features.append({'input_ids':cur_ids, 'input_tok2word':cur_tok2word,
                'input_tags':None, 'refs':None, 'turn':None})
    batches = asa_datastream.make_batch(features, 'sentiment', args.batch_size, is_sort=False, is_shuffle=False)

    sentiments = defaultdict(list)
    turn_id = 0
    for ori_batch in batches:
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        batch_outputs = sentiment_model(batch)
        batch_wordseq_lengths = batch_outputs['wordseq_lengths'].cpu().tolist() # [batch]
        for i, tag_ids in enumerate(batch_outputs['predictions'].cpu().tolist()): # [batch, wordseq]
            wordseq_len = batch_wordseq_lengths[i]
            for senti in asa_evaluator.extract_sentiment_from_tags(tag_ids[:wordseq_len]):
                st, ed, x = senti
                sentiments[turn_id].append({'variables':None, 'turn_id':turn_id, 'span':(st,ed), 'senti':x})
            turn_id += 1
    assert len(dialogue['conv']) == turn_id
    n_senti = sum(len(x) for x in sentiments.values())

    # decode mention
    features = []
    all_ids = []
    all_tok2word = []
    all_offsets = [0,]
    all_lex = []
    all_sentid = [] # start from 1 to avoid padding 0s
    for i, turn in enumerate(dialogue['conv']):
        cur_ids, cur_tok2word = asa_datastream.bert_tokenize(turn, tokenizer, args.tok2word_strategy) # [tok_seq], [word_seq, word_len]
        asa_datastream.merge(all_ids, all_tok2word, cur_ids, cur_tok2word)
        all_offsets.append(len(all_tok2word))
        all_lex.extend(turn)
        all_sentid.extend([i+1 for _ in turn])
        for senti in sentiments[i]:
            # ADD w_1^1, ..., w_1^{N_1}, ..., w_i^{N_i} [SEP]
            input_ids = all_ids + [SEP_ID,]
            input_tok2word = all_tok2word + [[len(input_ids)-1]]
            input_sentid = all_sentid + [i+2,]
            input_senti_mask = [0.0 for _ in input_tok2word]
            input_content_bound = len(all_tok2word)-1
            # ADD w_{s_j}^1, ..., w_{s_j}^{|s_j|}
            senti_st, senti_ed = senti['span'] # [st, ed]
            senti_ids, senti_tok2word = asa_datastream.bert_tokenize(turn[senti_st:senti_ed+1], tokenizer, args.tok2word_strategy)
            asa_datastream.merge(input_ids, input_tok2word, senti_ids, senti_tok2word)
            input_sentid.extend([i+3 for _ in senti_tok2word])
            input_senti_mask.extend([1.0 for _ in senti_tok2word])
            # ADD [CLS]
            input_ids += [CLS_ID,]
            input_tok2word += [[len(input_ids)-1]]
            input_sentid.append(i+4)
            input_senti_mask.append(0.0)
            senti_lex = ' '.join(turn[senti_st:senti_ed+1])
            features.append({'input_ids':input_ids, 'input_tok2word':input_tok2word, 'input_sentid':input_sentid,
                'input_senti_mask':input_senti_mask, 'input_content_bound':input_content_bound, 'input_ref':None,
                'refs':None, 'all_lex':all_lex, 'senti_lex':senti_lex, 'is_cross':None})
    batches = asa_datastream.make_batch(features, 'mention', args.batch_size, is_sort=False, is_shuffle=False)
    assert len(features) == n_senti

    mentions = []
    for step, ori_batch in enumerate(batches):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        batch_outputs = mention_model(batch)
        batch_wordseq_maxlen = batch_outputs['wordseq_num']
        for i, x in enumerate(batch_outputs['predictions'].cpu().tolist()):
            st = x // batch_wordseq_maxlen
            ed = x % batch_wordseq_maxlen
            turn_id = 0
            while all_offsets[turn_id+1] <= st:
                turn_id += 1
            turn_st = st - all_offsets[turn_id]
            turn_ed = ed - all_offsets[turn_id]
            assert turn_st >= 0
            mentions.append({'turn_id':turn_id, 'span':(turn_st,turn_ed)})
    assert len(mentions) == n_senti

    sentiments_list = []
    for i in range(len(dialogue['conv'])):
        sentiments_list.extend(sentiments[i])
    assert len(sentiments_list) == n_senti

    return sentiments_list, mentions


def merge_results(dialogue, sentiments, mentions):
    dialogue['casa_st'] = []
    dialogue['casa_ed'] = []
    senti_occupy = []
    for i in range(len(dialogue['conv'])):
        dialogue['casa_st'].append(['',]*len(dialogue['conv'][i]))
        dialogue['casa_ed'].append(['',]*len(dialogue['conv'][i]))
        senti_occupy.append([False,]*len(dialogue['conv'][i]))

    for senti in sentiments:
        senti_st, senti_ed = senti['span']
        senti_tid = senti['turn_id']
        for j in range(senti_st, senti_ed+1):
            assert senti_occupy[senti_tid][j] == False
            senti_occupy[senti_tid][j] = True

    varlist = list(string.ascii_lowercase)
    x_map = {1:'+1', 0:'0', -1:'-1'}
    vid = 0
    for i, (senti, mentn) in enumerate(zip(sentiments, mentions)):
        senti_st, senti_ed = senti['span']
        senti_tid = senti['turn_id']
        senti_x = x_map[senti['senti']]
        mentn_st, mentn_ed = mentn['span']
        mentn_tid = mentn['turn_id']

        conflict = False
        for j in range(mentn_st, mentn_ed+1):
            conflict |= senti_occupy[mentn_tid][j]
        if conflict:
            continue

        var = varlist[vid]
        vid += 1
        dialogue['casa_st'][senti_tid][senti_st] = '[{}'.format(var)
        dialogue['casa_ed'][senti_tid][senti_ed] = '{}##{}]'.format(senti_x, var)

        if dialogue['casa_st'][mentn_tid][mentn_st] == '':
            dialogue['casa_st'][mentn_tid][mentn_st] = '[{}'.format(var)
        else:
            dialogue['casa_st'][mentn_tid][mentn_st] = dialogue['casa_st'][mentn_tid][mentn_st] + '+{}'.format(var)

        if dialogue['casa_ed'][mentn_tid][mentn_ed] == '':
            dialogue['casa_ed'][mentn_tid][mentn_ed] = '{}]'.format(var)
        else:
            dialogue['casa_ed'][mentn_tid][mentn_ed] = '{}+'.format(var) + dialogue['casa_ed'][mentn_tid][mentn_ed]

    turns_with_casa = []
    for case_st, turn, casa_ed in zip(dialogue['casa_st'], dialogue['conv'], dialogue['casa_ed']):
        twc = []
        for st, x, ed in zip(case_st, turn, casa_ed):
            if st == '' and ed == '':
                twc.append(x)
            elif st != '' and ed != '':
                twc.append(' '.join([st,x,ed]))
            else:
                tmp = [st,x] if ed == '' else [x,ed]
                twc.append(' '.join(tmp))
        turns_with_casa.append(' '.join(twc))
    return '\n'.join(turns_with_casa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Should be consistent with training')
    parser.add_argument('--cuda_device', type=str, required=True, help='GPU ids (e.g. "1" or "1,2")')
    parser.add_argument('--bert_version', type=str, required=True, help='BERT version (e.g. "bert-base-chinese"')
    parser.add_argument('--tok2word_strategy', type=str, required=True, help='Should be consistent with training, e.g. avg')
    parser.add_argument('--mention_model_path', type=str, required=True, help='The saved mention model')
    parser.add_argument('--sentiment_model_path', type=str, required=True, help='The saved sentiment model')
    parser.add_argument('--in_path', type=str, default=None, help='Path to the input file')
    parser.add_argument('--out_path', type=str, default=None, help='Path to the output file')
    args, unparsed = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}'.format(device, n_gpu))

    tokenizer = BertTokenizer.from_pretrained(args.bert_version)

    print('Compiling model')
    mention_model = asa_model.BertAsaMe.from_pretrained(args.bert_version)
    mention_model.load_state_dict(torch.load(args.mention_model_path))
    mention_model.to(device)
    mention_model.eval()

    sentiment_model = asa_model.BertAsaSe.from_pretrained(args.bert_version)
    sentiment_model.load_state_dict(torch.load(args.sentiment_model_path))
    sentiment_model.to(device)
    sentiment_model.eval()

    print('Loading data')
    data = asa_datastream.load_data(args.in_path)
    print("Num dialogues = {}".format(len(data)))

    print('Decoding')
    f = open(args.out_path, 'w')
    for dialogue in data:
        sentiments, mentions = decode_dialogue(args, dialogue, sentiment_model, mention_model, tokenizer)
        outputs = merge_results(dialogue, sentiments, mentions)
        f.write(outputs+'\n\n')
    f.close()


