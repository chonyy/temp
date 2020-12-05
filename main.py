import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding, read_corpus_custom

def FormatOutput(y_pred, testdata_list, testdata_article_id_list):
    r"""
    Format data.
    """
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id, article_pred in enumerate(y_pred):
        pos=0
        start_pos=None
        end_pos=None
        entity_text=None
        entity_type=None
        # print(article_pred)
        for pred_id, pred in enumerate(article_pred):
            # print('pred', pred)
            if pred[0]=='B':
                start_pos=pos
                entity_type=pred[2:]
            elif start_pos is not None and pred[0]=='I' and pred_id == len(article_pred) - 1:
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            elif start_pos is not None and pred[0]=='I' and article_pred[pred_id+1][0]=='O':
                end_pos=pos
                entity_text=''.join([testdata_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
                line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output+=line+'\n'
            
            pos+=1     
    output_path = 'output.tsv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output

def readToData2(path):
    lines = None
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    data = []
    data_tmp = []
    for idx, l in enumerate(lines):
    
        if(l == None or len(l) == 1):
            continue
        elif(l.startswith('article')):
            data_tmp = []
        elif(l.startswith('-----')):
            data.append(data_tmp)
        else:
            for c in l:
                data_tmp.append(c)
    return data

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=32, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=500, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


## get char embeddings
# word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
word2id = read_dictionary('word.pkl')
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'demo':
    # train_path = os.path.join('.', args.train_data, 'train_data')
    # test_path = os.path.join('.', args.test_data, 'test_data')
    train_path = 'sample3.data'
    test_path = 'train.data'
    train_data = read_corpus_custom(train_path)
    # print(train_data)
    test_data = read_corpus_custom(test_path); test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        dataList = readToData2('dev.txt')
        answer = []
        for idx, article in enumerate(dataList):
            demo_sent = article
            demo_data = [(demo_sent, ['O'] * len(demo_sent))]
            tag = model.demo_one(sess, demo_data)
            # print(tag)
            print(idx, len(tag))
            answer.append(tag)

        print(len(answer))
        print(len(answer[0]))
        print(len(dataList))
        print(len(dataList[0]))
        FormatOutput(answer, dataList, np.arange(len(dataList)))

        # while(1):
        #     print('Please input your sentence:')
        #     demo_sent = input()
        #     if demo_sent == '' or demo_sent.isspace():
        #         print('See you next time!')
        #         break
        #     else:
        #         demo_sent = list(demo_sent.strip())
        #         demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        #         print(len(demo_data))
        #         print(demo_data)
        #         tag = model.demo_one(sess, demo_data)
        #         print(tag)
        #         # PER, LOC, ORG = get_entity(tag, demo_sent)
        #         TIME = get_entity(tag, demo_sent)
        #         # print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
        #         print('TIME: {}'.format(TIME))


