
import os, sys, json


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def load(path):
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
                if len(tok) > 1 and is_int(tok[:-1]) == False: # for situations like '》]'
                    assert False, line # right now make sure this doesn't happen
                    #turn.append(tok[:-1])
                    #tok = tok[-1]
                ed = len(turn) - 1 # [st, ed]
                senti = None if tok == ']' else int(tok[:-1])
                if senti is not None:
                    sentiment.append({'variables':variables, 'span':(st,ed), 'turn_id':len(conv), 'senti':senti})
                else:
                    assert len(variables) == 1, line
                    mention.append({'var':variables[0], 'span':(st,ed), 'turn_id':len(conv)})
            else:
                turn.append(tok)
        conv.append(turn)

    sentiments = []
    mentions = []
    for instance in data:
        conv = instance['conv']
        for senti in instance['sentiment']:
            tid = senti['turn_id']
            st, ed = senti['span']
            sentiments.append(' '.join(conv[tid][st:ed+1]))
        for mentn in instance['mention']:
            tid = mentn['turn_id']
            st, ed = mentn['span']
            mentions.append(' '.join(conv[tid][st:ed+1]))
    return sentiments, mentions


def oov_rate(dct, lst):
    right = sum(x not in dct for x in lst)
    total = len(lst)
    return 1.0*right/total


train_sentiments, train_mentions = load('duconv_train.txt')
train_sentiments = set(train_sentiments)
train_mentions = set(train_mentions)

test_sentiments, test_mentions = load('duconv_test.txt')
print('DuConv test mention: {:.4f}, sentiment: {:.4f}'.format(oov_rate(train_mentions, test_mentions),
    oov_rate(train_sentiments, test_sentiments)))

news_sentiments, news_mentions = load('news_dialog_dialogue.txt')
print('NewsDialog mention: {:.4f}, sentiment: {:.4f}'.format(oov_rate(train_mentions, news_mentions),
    oov_rate(train_sentiments, news_sentiments)))

