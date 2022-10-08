
import os, sys, json

def remove_space(sent):
    is_last_ascii = False
    new_sent = []
    for tok in sent.split():
        if len(new_sent) == 0:
            new_sent.append(tok)
        elif is_last_ascii and tok.isascii():
            new_sent.append(tok)
        else:
            new_sent[-1] = new_sent[-1] + tok
        is_last_ascii = tok.isascii()
    return ' '.join(new_sent)

def process_file(inpath, outpath, start_id):
    with open(inpath, 'r') as infile, open(outpath, 'w') as outfile:
        id = start_id
        conv = []
        for line in infile:
            if line.strip() == '':
                if len(conv) > 0:
                    txt = '\n'.join(conv)
                    obj = {'id': id, 'txt': txt}
                    outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    id += 1
                    conv = []
            else:
                conv.append(line.strip())
    if len(conv) > 0:
        txt = '\n'.join(conv)
        obj = {'id': id, 'txt': txt}
        outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

process_file('trial_rewrite.txt', 'trial_rewrite.jsonl', 0)
