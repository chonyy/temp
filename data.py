import sys, pickle, os, random
import numpy as np

## tags, BIO
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6
#              }

tag2label = {"O": 0,
             "B-name": 1, "I-name": 2,
             "B-location": 3, "I-location": 4,
             "B-time": 5, "I-time": 6,
             "B-contact": 7, "I-contact": 8,
             "B-ID": 9, "I-ID": 10,
             "B-profession": 11, "I-profession": 12,
             "B-biomarker": 13, "I-biomarker": 14,
             "B-family": 15, "I-family": 16,
             "B-clinical_event": 17, "I-clinical_event": 18,
             "B-special_skills": 19, "I-special_skills": 20,
             "B-unique_treatment": 21, "I-unique_treatment": 22,
             "B-account":23, "I-account": 24,
             "B-organization": 25, "I-organization": 26, 
             "B-education": 27, "I-education": 28,
             "B-money": 29, "I-money": 30,
             "B-belonging_mark": 31, "I-belonging_mark": 32,
             "B-med_exam": 33, "I-med_exam": 34,
             "B-others": 35, "I-others": 36
             }

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            # print('char', char)
            # print('label', label)
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

def read_corpus_custom(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for idx, line in enumerate(lines):
        if line == '\n':
            continue
        if line[0] != '：':
            [char, label] = line.strip().split()
            # print('char', char)
            # print('label', label)
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

def read_corpus_custom_split(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for idx, line in enumerate(lines):
        if line == '\n':
            continue
        if(line[0] == '：'):
            sent_, tag_ = [], []
        elif (line[0] == '！' or line[0] == '？' or line[0] == '。'):
            # print(len(sent_))
            data.append((sent_, tag_))
            sent_, tag_ = [], []
        else:
            [char, label] = line.strip().split()
            # print('char', char)
            # print('label', label)
            sent_.append(char)
            tag_.append(label)
    if(([], []) in data):
        data.remove(([], []))
    return data

def read_corpus_custom_whole(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for idx, line in enumerate(lines):
        if line == '\n':
            continue
        if line[0] != '：':
            [char, label] = line.strip().split()
            # print('char', char)
            # print('label', label)
            sent_.append(char)
            tag_.append(label)
        else:
            # print(len(sent_) - 2)
            data.append((sent_[:-2], tag_[:-2]))
            sent_, tag_ = [], []
    if(([], []) in data):
        data.remove(([], []))
    return data

def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

