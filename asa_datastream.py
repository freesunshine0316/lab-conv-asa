
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


def load_and_extract_features(path, tokenizer, tok2word_strategy, task):
    data = []
    conv, sentiment, mention = [], [], []
    for i, line in enumerate(open(path, 'r')):
        if line.strip() == '': # end of a dialogue
            if len(conv) > 0:
                data.append({'conv':conv, 'sentiment':sentiment, 'mention':mention})
            conv, sentiment, mention = [], [], []
            continue
        turn = []
        for tok in line.split():
            if tok.startswith('['):
                st = len(turn)
                variables = tok[1:].split('+')
            elif tok.endswith(']'):
                if len(tok) > 1 and is_int(tok[:-1]) == False: # handling situations like '》]'
                    assert False, line
                    #turn.append(tok[:-1])
                    #tok = tok[-1]
                ed = len(turn) - 1 # [st, ed]
                senti = None if tok == ']' else int(tok[:-1])
                if senti is not None:
                    for var in variables:
                        sentiment.append({'var':var, 'span':(st,ed), 'turn_id':len(conv), 'senti':senti})
                    #print('{} ||| {}'.format(i, ' '.join(turn[st:ed+1])))
                else:
                    assert len(variables) == 1
                    var = variables[0]
                    mention.append({'var':var, 'span':(st,ed), 'turn_id':len(conv)})
            else:
                turn.append(tok)
        conv.append(turn)

    if task == 'sentiment':
        return extract_features_sentiment(data, tokenizer, tok2word_strategy)
    elif task == 'mention':
        return extract_features_mention(data, tokenizer, tok2word_strategy)
    else:
        assert False, 'Unsupported task: ' + task


def bert_tokenize(word_seq, tokenizer, tok2word_strategy):
    input_ids = [] # [tok number]
    input_tok2word = [] # [word number, word length]
    total_offset = 0
    for word in word_seq:
        if word in ('A:', 'B:'):
            toks = ['<S>',] if word == 'A:' else ['<T>',]
        else:
            toks = [x if x in tokenizer.vocab else '[UNK]' for x in tokenizer.tokenize(word)]
        assert len(toks) > 0
        idxs = tokenizer.convert_tokens_to_ids(toks)
        input_ids.extend(idxs)
        positions = [i + total_offset for i in range(len(idxs))]
        total_offset += len(idxs)
        if tok2word_strategy == 'first':
            input_tok2word.append(positions[:1])
        elif tok2word_strategy == 'last':
            input_tok2word.append(positions[-1:])
        elif tok2word_strategy == 'avg':
            input_tok2word.append(positions)
        else:
            assert False, 'Unsupported tok2word_strategy: ' + tok2word_strategy
    return input_ids, input_tok2word


def extract_features_sentiment(data, tokenizer, tok2word_strategy):
    features = []
    #right, total = 0.0, 0.0
    for dialogue in data:
        for i, turn in enumerate(dialogue['conv']):
            input_ids, input_tok2word = bert_tokenize(turn, tokenizer, tok2word_strategy) # [tok_seq], [word_seq, word_len]
            input_tags = [TAG_MAPPING['O'] for _ in input_tok2word] # [word_seq]
            refs = set()
            for senti in dialogue['sentiment']:
                if senti['turn_id'] == i:
                    st, ed = senti['span']
                    refs.add((st, ed, senti['senti']))
                    senti_str = SENTI_STR_MAPPING[senti['senti']]
                    input_tags[st] = TAG_MAPPING[senti_str+'B']
                    for j in range(st+1, ed+1):
                        input_tags[j] = TAG_MAPPING[senti_str+'I']
            features.append({'input_ids':input_ids, 'input_tok2word':input_tok2word, 'input_tags':input_tags, 'refs':refs})
            #right += sum(x != 0 for x in input_tags)
            #total += len(input_tags)
    #print('Sentiment tags percent: %.2f' % (100*right/total))
    return features


# a_ids: [232, 897, 23], a_tok2word: [[0, 1], [2]]
# b_ids: [213, 1324, 242, 212], b_tok2word: [[0, 1], [2, 3]]
# ===> a_ids: [232, 897, 23, 213, 1324, 242, 212], a_tok2word: [[0, 1], [2], [3, 4], [5, 6]]
def merge(a_ids, a_tok2word, b_ids, b_tok2word):
    offset = len(a_ids)
    a_ids += b_ids
    for t2w in b_tok2word:
        t2w = [x + offset for x in t2w]
        a_tok2word.append(t2w)


# w_1^1, ..., w_1^{N_1}, ..., w_i^{N_i} [SEP] w_{s_j}^1, ..., w_{s_j}^{|s_j|} [CLS]
def extract_features_mention(data, tokenizer, tok2word_strategy):
    CLS_ID, SEP_ID = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    features = []
    for dialogue in data:
        all_ids = []
        all_tok2word = []
        all_offsets = [0,]
        for i, turn in enumerate(dialogue['conv']):
            cur_ids, cur_tok2word = bert_tokenize(turn, tokenizer, tok2word_strategy) # [tok_seq], [word_seq, word_len]
            merge(all_ids, all_tok2word, cur_ids, cur_tok2word)
            all_offsets.append(len(all_tok2word))
            for senti in dialogue['sentiment']:
                if senti['turn_id'] == i:
                    has_mention = False

                    # ADD w_1^1, ..., w_1^{N_1}, ..., w_i^{N_i} [SEP]
                    input_ids = all_ids + [SEP_ID,]
                    input_tok2word = all_tok2word + [[len(input_ids)-1]]
                    input_senti_mask = [0.0 for _ in input_tok2word]
                    input_content_bound = len(all_tok2word)-1

                    # ADD w_{s_j}^1, ..., w_{s_j}^{|s_j|} [CLS]
                    senti_st, senti_ed = senti['span'] # [st, ed)
                    senti_ids, senti_tok2word = bert_tokenize(turn[senti_st:senti_ed+1], tokenizer, tok2word_strategy)
                    merge(input_ids, input_tok2word, senti_ids, senti_tok2word)
                    input_senti_mask.extend([1.0 for _ in senti_tok2word])
                    input_ids += [CLS_ID,]
                    input_tok2word += [[len(input_ids)-1]]
                    input_senti_mask.append(0.0)

                    input_ref = []
                    var = senti['var']
                    for mentn in dialogue['mention']:
                        if mentn['var'] == var and mentn['turn_id'] <= i:
                            has_mention = True
                            st, ed = mentn['span']
                            st, ed = st + all_offsets[mentn['turn_id']], ed + all_offsets[mentn['turn_id']]
                            input_ref.append((st,ed))
                    if has_mention:
                        features.append({'input_ids':input_ids, 'input_tok2word':input_tok2word, 'input_ref':input_ref,
                            'input_senti_mask':input_senti_mask, 'input_content_bound':input_content_bound})
    return features


def make_batch(features, task, batch_size, is_sort=True, is_shuffle=False):
    if task == 'sentiment':
        return make_batch_sentiment(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)
    elif task == 'mention':
        return make_batch_mention(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)
    else:
        assert False, 'Unknown'


def make_batch_unified(features, B, N):
    maxseq, maxwordseq, maxwordlen = 0, 0, 0
    for i in range(0, B):
        maxseq = max(maxseq, len(features[N+i]['input_ids']))
        maxwordseq = max(maxwordseq, len(features[N+i]['input_tok2word']))
        for x in features[N+i]['input_tok2word']:
            maxwordlen = max(maxwordlen, len(x))

    input_ids = np.zeros([B, maxseq], dtype=np.long)
    input_mask = np.zeros([B, maxseq], dtype=np.float)
    input_tok2word = np.zeros([B, maxwordseq, maxwordlen], dtype=np.long)
    input_tok2word_mask = np.zeros([B, maxwordseq, maxwordlen], dtype=np.float)

    for i in range(0, B):
        curseq = len(features[N+i]['input_ids'])
        curwordseq = len(features[N+i]['input_tok2word'])
        input_ids[i,:curseq] = features[N+i]['input_ids']
        input_mask[i,:curseq] = [1,]*curseq
        for j in range(0, curwordseq):
            curwordlen = len(features[N+i]['input_tok2word'][j])
            input_tok2word[i,j,:curwordlen] = features[N+i]['input_tok2word'][j]
            input_tok2word_mask[i,j,:curwordlen] = [1,]*curwordlen

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    input_tok2word = torch.tensor(input_tok2word, dtype=torch.long)
    input_tok2word_mask = torch.tensor(input_tok2word_mask, dtype=torch.float)

    return {'input_ids':input_ids, 'input_mask':input_mask, 'input_tok2word':input_tok2word,
            'input_tok2word_mask':input_tok2word_mask}, maxseq, maxwordseq, maxwordlen


def make_batch_sentiment(features, batch_size, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        batch, maxseq, maxwordseq, maxwordlen = make_batch_unified(features, B, N)
        input_tags = np.zeros([B, maxwordseq], dtype=np.long)
        for i in range(0, B):
            curwordseq = len(features[N+i]['input_tok2word'])
            input_tags[i,:curwordseq] = features[N+i]['input_tags']
        batch['input_tags'] = torch.tensor(input_tags, dtype=torch.long)
        batch['refs'] = [features[N+i]['refs'] for i in range(0, B)]
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
        batch, maxseq, maxwordseq, maxwordlen = make_batch_unified(features, B, N)
        input_senti_mask = np.zeros([B, maxwordseq], dtype=np.float)
        input_content_mask = np.zeros([B, maxwordseq], dtype=np.float)
        input_ref = np.zeros([B, maxwordseq, 2], dtype=np.float)
        refs = [set() for i in range(0, B)]
        for i in range(0, B):
            curwordseq = len(features[N+i]['input_tok2word'])
            input_senti_mask[i,:curwordseq] = features[N+i]['input_senti_mask']
            curcontent = features[N+i]['input_content_bound']
            input_content_mask[i,:curcontent] = [1.0,]*curcontent
            ref_num = len(features[N+i]['input_ref'])
            assert ref_num > 0
            for st, ed in features[N+i]['input_ref']:
                input_ref[i,st,0] = 1.0/ref_num
                input_ref[i,ed,1] = 1.0/ref_num
                refs[i].add((st,ed))
        batch['input_senti_mask'] = torch.tensor(input_senti_mask, dtype=torch.float)
        batch['input_content_mask'] = torch.tensor(input_content_mask, dtype=torch.float)
        batch['input_ref'] = torch.tensor(input_ref, dtype=torch.float)
        batch['refs'] = refs
        batches.append(batch)
        N += B
    return batches



