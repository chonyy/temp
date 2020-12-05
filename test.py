import pickle

# def read_corpus(corpus_path):
#     """
#     read corpus and return the list of samples
#     :param corpus_path:
#     :return: data
#     """
#     data = []
#     with open(corpus_path, encoding='utf-8') as fr:
#         lines = fr.readlines()
#     sent_, tag_ = [], []
#     for line in lines:
#         if line != '\n':
#             [char, label] = line.strip().split()
#             sent_.append(char)
#             tag_.append(label)
#         else:
#             data.append((sent_, tag_))
#             sent_, tag_ = [], []

#     return data

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

    print(word2id)
    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

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

data = read_corpus_custom_whole('train.data')
print(data[-5:])

# vocab_build('word2.pkl', 'sample3.data', 3)

# data = read_corpus('data_path/train_data')
# data = read_corpus('train.data')
# print(len(data))
# print(type(data))
# for t in data:
#     print(len(t[0]))