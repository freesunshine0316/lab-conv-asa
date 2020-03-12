
import os, sys, json

def load(path):
    data = []
    conv = []
    for i, line in enumerate(open(path, 'r')):
        if line.strip() == '': # end of a dialogue
            if len(conv) > 0:
                data.append(conv)
                conv = []
        else:
            conv.append(line.strip())
    if len(conv) > 0:
        data.append(conv)
    return data


data = load(sys.argv[1]+'.txt')
print(len(data))
prev = 0
for subset, portion in zip(['dev', 'test', 'train'], [0.1, 0.1, 0.8]):
    n = len(data) - prev if subset == 'train' else int(len(data) * portion)
    f = open(sys.argv[1]+'_'+subset+'.txt', 'w')
    print('{} {}'.format(subset, n))
    for i in range(prev, prev+n):
        for sent in data[i]:
            f.write(sent+'\n')
        f.write('\n')
    prev += n

