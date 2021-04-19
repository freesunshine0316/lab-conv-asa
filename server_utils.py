import codecs
import json

def preprocess_turns(turn_words):
    """
    this function is used to rewrite the turns by adding the speakers (human or agent), return the index mapping (from original word index to new word index)
    :param turn_words:
    :return:
    """

    res = []
    idx_mapping = {}
    new_index = -1
    old_index = -1

    # we need to add "B" into the words. This is because in some cases, the first utterance may include the reference to B
    res.append("agent")
    new_index += 1

    for turn_index in range(0, len(turn_words)):
        if turn_index % 2 == 0:
            res.append("human")
        else:
            res.append("agent")
        new_index += 1

        words = turn_words[turn_index]
        for word in words:
            res.append(word)
            old_index += 1
            new_index += 1
            idx_mapping[old_index] = new_index
    return res, idx_mapping

def load_file(path):
    with codecs.open(path, 'r', 'utf-8') as fr:
        res = []
        for line in fr:
            info = line.split("|||")
            sent_id = info[0]
            pred_idx = int(info[1].strip())
            words = info[2].strip().split(" ")
            dialog_turns = [int(x) for x in info[3].split(" ")]
            tags = info[4].split(" ")
            res.append((sent_id, pred_idx, words, dialog_turns, tags))
        return res

def load_vocab(path, symbol_idx=None):
    if symbol_idx is None:
        symbol_idx = {}
    with codecs.open(path, 'r', "utf-8") as fr:
        for symbol in fr:
            symbol = symbol.strip()
            if symbol not in symbol_idx:
                symbol_idx[symbol] = len(symbol_idx)
    return symbol_idx

def bert_vectorize_data(tokenizer, data, label_vocab, seg_type_id_map, mode="Train"):
    if mode == "Train":
        sent_id, pred_idx, words, dialog_turns, labels = data
        label_vec = []
        labels.append("[SEP]")
        labels.insert(0, "[CLS]")
    else:
        pred_idx, words, dialog_turns = data

    segment_ids = []
    words.append("[SEP]")
    dialog_turns.append("100")

    pred_idx += 1
    words.insert(0, "[CLS]")
    dialog_turns.insert(0, "100")

    bert_word_len = []
    sen_vec = []
    pred_vec = []
    dialog_turn_vec = []
    seg_type = "agent"
    for _ in range(len(words)):
        if words[_] == "agent":
            seg_type = "agent"
        elif words[_] == "human":
            seg_type = "human"

        if words[_] == "[CLS]" or words[_] == "[SEP]":
            tokens = [words[_]]
            seg_type = words[_]
        else:
            tokens = tokenizer.tokenize(words[_])

        bert_word_len.append(len(tokens))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        sen_vec.extend(ids)
        # segment_ids.extend([seg_type_id_map[seg_type]] * len(ids))
        segment_ids.extend([0] * len(ids))

        assert len(sen_vec) == len(segment_ids)

        if mode == "Train":
            if labels[_] in label_vocab:
                label_vec.append(label_vocab[labels[_]])
                if labels[_] == "O":
                    label_vec.extend([label_vocab["O"]] * (len(ids) - 1))  # assign label "X" for sub-word
                elif len(ids) > 1:
                    label_vec.extend((len(ids) - 1) * [label_vocab["I" + labels[_][1:]]])
            else:
                label_vec.extend([label_vocab["O"]] * len(ids))

            assert len(sen_vec) == len(label_vec)

        if _ == pred_idx:
            pred_vec.extend([2] * len(ids))
        else:
            pred_vec.extend([1] * len(ids))

        dialog_turn = int(dialog_turns[_])
        if dialog_turn == 100:
            dialog_turn_vec.extend([0] * len(ids))
        else:
            dialog_turn = min(9, dialog_turn + 1)
            dialog_turn_vec.extend([dialog_turn] * len(ids))

    if mode == "Train":
        return sen_vec, segment_ids, pred_vec, dialog_turn_vec, label_vec
    else:
        return sen_vec, segment_ids, pred_vec, dialog_turn_vec, bert_word_len


# def entity_tagging(words, entity_mention_hash, entity_mention_set):
#     tags = ['O'] * len(words)
#     curEntity = ""
#     for cur_idx in range(len(words)):
#         if words[cur_idx] == "human" or words[cur_idx] == "agent":
#             curEntity = ""
#             continue
#         cur_word = curEntity + words[cur_idx]
#         if cur_word in entity_mention_hash:
#             if cur_word in entity_mention_set:
#                 entity_len = len(cur_word.split(" "))
#                 for i in range(cur_idx - entity_len + 2, cur_idx + 1):
#                     tags[i] = "I-E"
#                 tags[cur_idx + 1 - entity_len] = "B-E"
#
#             curEntity += words[cur_idx] + " "
#         else:
#             curEntity = ""
#
#     return tags


# if __name__ == "__main__":
#     res = load_file("data/train.txt")
#     word_vocab = set()
#     tag_vocab = set()
#     for pred_idx, words, tags in res:
#         for word in words:
#             if word.strip() != "" and word not in word_vocab:
#                 word_vocab.add(word)
#         for tag in tags:
#             if tag not in tag_vocab:
#                 tag_vocab.add(tag)
#     word_vocab = list(word_vocab)
#     tag_vocab = list(tag_vocab)
#     word_vocab.insert(0, "<unk>")
#     word_vocab.insert(0, "<pad>")
#     print("there are {} words, {} tags".format(len(word_vocab), len(tag_vocab)))
#     with codecs.open("data/vocab.txt", 'w', 'utf-8') as fw:
#         for word in word_vocab:
#             fw.write(word + "\n")
#     with codecs.open("data/label.txt", "w", "utf-8") as fw:
#         for label in tag_vocab:
#             fw.write(label + "\n")
