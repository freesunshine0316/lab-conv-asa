
import os, sys, json, codecs
import numpy as np
import torch
import ast
import collections

SENTI_STR_MAPPING = {1:'Pos', 0:'Neu', -1:'Neg'}
TAG_MAPPING = {'O':0, 'PosB':1, 'PosI':2, 'NeuB':3, 'NeuI':4, 'NegB':5, 'NegI':6}
TAGS = ['O', 'PosB', 'PosI', 'NeuB', 'NeuI', 'NegB', 'NegI']


def is_tag_begin(tid):
    return tid in (1,3,5,)


def is_tag_inner(tid):
    return tid in (2,4,6,)


def check_tag_sentiment(tid):
    if tid in (1,2,):
        return 1
    elif tid in (3,4,):
        return 0
    elif tid in (5,6,):
        return -1
    assert False, 'illegal tid {}'.format(tid)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_data(path, tokenizer):
    data = []
    conv, sentiment, mention = [], [], []
    for i, line in enumerate(open(path, 'r')):
        if line.strip() == '': # end of a dialogue
            if len(conv) > 0:
                data.append({'conv':conv, 'sentiment':sentiment, 'mention':mention})
                #for x in data[-1]['mention']:
                #    tid = x['turn_id']
                #    st, ed = x['span']
                #    print(tokenizer.decode(data[-1]['conv'][tid][st:ed+1]))
            conv, sentiment, mention = [], [], []
        else:
            turn = []
            flag = False
            for tok in line.split():
                if tok.startswith('['):
                    assert flag == False, 'Erroneous CASA annotations for {}'.format(line)
                    st = len(turn)
                    variables = tok[1:].split('+')
                    flag = True
                elif tok.endswith(']') and (tok == ']' or is_int(tok[:-1])):
                    assert flag == True, 'Erroneous CASA annotations for {}'.format(line)
                    ed = len(turn) - 1 # [st, ed]
                    senti = None if tok == ']' else int(tok[:-1])
                    if senti != None: # sentiment
                        assert senti in (-1, 0, 1)
                        sentiment.append({'variables':variables, 'span':(st,ed), 'turn_id':len(conv), 'senti':senti})
                    else: # mention
                        assert len(variables) == 1, 'Erroneous CASA annotations for {}'.format(line)
                        mention.append({'var':variables[0], 'span':(st,ed), 'turn_id':len(conv)})
                    flag = False
                else:
                    turn.extend(tokenizer.encode(tok, add_special_tokens=False))
            conv.append(turn)
    if len(conv) > 0:
        data.append({'conv':conv, 'sentiment':sentiment, 'mention':mention})
        #for x in data[-1]['mention']:
        #    tid = x['turn_id']
        #    st, ed = x['span']
        #    print(tokenizer.decode(data[-1]['conv'][tid][st:ed+1]))
    return data


def load_and_extract_features(path, tokenizer, task, portion="all"):
    data = load_data(path, tokenizer)
    num_conv, num_sentence, num_mntn = 0.0, 0.0, 0.0
    num_senti, num_senti_cross, num_senti_pos, num_senti_neu, num_senti_neg = 0.0, 0.0, 0.0, 0.0, 0.0
    num_mntn_tokens = 0.0
    for instance in data:
        conv, sentiment, mention = instance['conv'], instance['sentiment'], instance['mention']
        num_conv += 1.0
        num_sentence += len(conv)
        num_senti += len(sentiment)
        for senti in sentiment:
            is_cross = not any(senti['turn_id'] == x['turn_id'] and x['var'] in senti['variables'] for x in mention)
            senti['is_cross'] = is_cross
            num_senti_cross += is_cross
            num_senti_pos += (senti['senti'] == 1)
            num_senti_neu += (senti['senti'] == 0)
            num_senti_neg += (senti['senti'] == -1)
        num_mntn += len(mention)
        num_mntn_tokens += sum([x['span'][1] - x['span'][0] + 1 for x in mention])
    print('Number of convs {} and sentences {}'.format(num_conv, num_sentence))
    print('Number of sentiments {}, cross {}, pos {}, neu {}, neg {}'.format(num_senti,
        num_senti_cross/num_senti, num_senti_pos/num_senti, num_senti_neu/num_senti, num_senti_neg/num_senti))
    print('Number of mention {}, avg tokens {:.2f}'.format(num_mntn, num_mntn_tokens/num_mntn))

    if task == 'sentiment':
        return extract_features_sentiment(data, tokenizer)
    elif task == 'mention':
        return extract_features_mention(data, tokenizer, portion)
    else:
        assert False, 'Unsupported task: ' + task


def extract_features_sentiment(data, tokenizer):
    #UNK_ID = tokenizer.unk_token_id
    features = []
    #unk_count, total_count = 0.0, 0.0
    for dialogue in data:
        for i, input_ids in enumerate(dialogue['conv']):
            input_tags = [TAG_MAPPING['O'] for _ in input_ids] # [seq]
            #unk_count += sum([x == UNK_ID for x in input_ids])
            #total_count += len(input_ids)
            refs = set()
            for senti in dialogue['sentiment']:
                if senti['turn_id'] == i:
                    st, ed = senti['span']
                    refs.add((st, ed, senti['senti']))
                    senti_str = SENTI_STR_MAPPING[senti['senti']]
                    input_tags[st] = TAG_MAPPING[senti_str+'B']
                    for j in range(st+1, ed+1):
                        input_tags[j] = TAG_MAPPING[senti_str+'I']
            features.append({'input_ids':input_ids, 'input_tags':input_tags, 'refs':refs})
    #print('UNK rate: {:.2f}'.format(100*unk_count/total_count))
    return features


# w_1^1, ..., w_1^{N_1}, ..., w_i^{N_i} [SEP] w_{s_j}^1, ..., w_{s_j}^{|s_j|}
def extract_features_mention(data, tokenizer, portion):
    CLS_ID, SEP_ID = tokenizer.cls_token_id, tokenizer.sep_token_id
    features = []
    for dialogue in data:
        all_ids = []
        all_offsets = [0,]
        all_sentids = [] # start from 1 to avoid padding 0s
        for i, cur_ids in enumerate(dialogue['conv']):
            all_ids += cur_ids
            all_offsets.append(all_offsets[-1]+len(cur_ids))
            all_sentids.extend([i+1 for _ in cur_ids])
            for senti in dialogue['sentiment']:
                if portion == 'cross' and senti['is_cross'] == False:
                    continue
                if portion == 'same' and senti['is_cross']:
                    continue

                if senti['turn_id'] == i:
                    has_mention = False

                    # ADD w_1^1, ..., w_1^{N_1}, ..., w_i^{N_i} [SEP]
                    input_ids = all_ids + [SEP_ID,]
                    input_sentids = all_sentids + [i+2,]
                    input_senti_mask = [0.0 for _ in input_ids]
                    input_content_bound = len(input_ids)-1

                    # ADD w_{s_j}^1, ..., w_{s_j}^{|s_j|}
                    senti_st, senti_ed = senti['span'] # [st, ed]
                    senti_ids = cur_ids[senti_st:senti_ed+1]
                    input_ids += senti_ids
                    input_sentids.extend([i+3 for _ in senti_ids])
                    input_senti_mask.extend([1.0 for _ in senti_ids])
                    #print(tokenizer.decode(senti_ids))

                    input_ref = []
                    refs = set()
                    for mentn in dialogue['mention']:
                        if mentn['var'] in senti['variables'] and mentn['turn_id'] <= i:
                            has_mention = True
                            st, ed = mentn['span']
                            #tid = mentn['turn_id']
                            #print('\t{}'.format(tokenizer.decode(dialogue['conv'][tid][st:ed+1])))
                            st, ed = st + all_offsets[mentn['turn_id']], ed + all_offsets[mentn['turn_id']]
                            input_ref.append((st,ed))
                            #print('\t{}'.format(tokenizer.decode(input_ids[st:ed+1])))
                            refs.add(tokenizer.decode(input_ids[st:ed+1]))
                    if has_mention:
                        features.append({'input_ids':input_ids, 'input_sentids':input_sentids, 'input_ref':input_ref,
                            'input_senti_mask':input_senti_mask, 'input_content_bound':input_content_bound, 'refs':refs,
                            'is_cross':senti['is_cross'], 'senti':senti['senti']})
    return features


def make_batch(features, task, batch_size, is_sort=True, is_shuffle=False):
    if task == 'sentiment':
        return make_batch_sentiment(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)
    elif task == 'mention':
        return make_batch_mention(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)
    else:
        assert False, 'Unknown'


def make_batch_unified(features, B, N):
    maxseq = 0
    for i in range(0, B):
        maxseq = max(maxseq, len(features[N+i]['input_ids']))

    input_ids = np.zeros([B, maxseq], dtype=np.long)
    input_mask = np.zeros([B, maxseq], dtype=np.float)

    for i in range(0, B):
        curseq = len(features[N+i]['input_ids'])
        input_ids[i,:curseq] = features[N+i]['input_ids']
        input_mask[i,:curseq] = [1,]*curseq

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    return {'input_ids':input_ids, 'input_mask':input_mask}, maxseq


def make_batch_sentiment(features, batch_size, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        batch, maxseq = make_batch_unified(features, B, N)
        if features[N]['input_tags'] != None:
            input_tags = np.zeros([B, maxseq], dtype=np.long)
            for i in range(0, B):
                curseq = len(features[N+i]['input_tags'])
                input_tags[i,:curseq] = features[N+i]['input_tags']
            batch['input_tags'] = torch.tensor(input_tags, dtype=torch.long)
            batch['refs'] = [features[N+i]['refs'] for i in range(0, B)]
        else:
            batch['input_tags'] = None
            batch['refs'] = None
        batches.append(batch)
        N += B
    return batches


def make_batch_mention(features, batch_size, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        batch, maxseq = make_batch_unified(features, B, N)
        input_sentids = np.zeros([B, maxseq], dtype=np.float)
        input_senti_mask = np.zeros([B, maxseq], dtype=np.float)
        input_content_mask = np.zeros([B, maxseq], dtype=np.float)
        input_ref = np.zeros([B, maxseq, 2], dtype=np.float)
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            input_sentids[i,:curseq] = features[N+i]['input_sentids']
            input_senti_mask[i,:curseq] = features[N+i]['input_senti_mask']
            bound = features[N+i]['input_content_bound']
            input_content_mask[i,:bound] = [1.0,] * bound
            if features[N]['input_ref'] != None:
                ref_num = len(features[N+i]['input_ref'])
                assert ref_num > 0
                for st, ed in features[N+i]['input_ref']:
                    input_ref[i,st,0] = 1.0/ref_num
                    input_ref[i,ed,1] = 1.0/ref_num
        batch['input_sentids'] = torch.tensor(input_sentids, dtype=torch.float)
        batch['input_senti_mask'] = torch.tensor(input_senti_mask, dtype=torch.float)
        batch['input_content_mask'] = torch.tensor(input_content_mask, dtype=torch.float)
        batch['senti'] = [features[N+i]['senti'] for i in range(0, B)]
        if features[N]['input_ref'] != None:
            batch['input_ref'] = torch.tensor(input_ref, dtype=torch.float)
            batch['refs'] = [features[N+i]['refs'] for i in range(0, B)]
            batch['is_cross'] = [features[N+i]['is_cross'] for i in range(0, B)]
        else:
            batch['input_ref'] = None
            batch['refs'] = None
            batch['is_cross'] = None
        batches.append(batch)
        N += B
    return batches



