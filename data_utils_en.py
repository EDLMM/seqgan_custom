# coding=utf-8
# import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize,RegexpTokenizer
import string

PAD_ID = 0
GO_ID = 1 
EOS_ID = 2
UNK_ID = 3
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
# max_vocabulary_size = 10000

def word_tokenize_no_punct(raw_str):
    moby_tokens = word_tokenize(raw_str)
    text_no_punct = [t for t in moby_tokens if t not in string.punctuation]
    return text_no_punct

def get_sent_list(file_path="./dataset/fake.csv",datatype="csv"):
    
    '''
    return [["word","word"...],
            ["word","word"...],
            ...]
    '''
    Row_list =[]
    if datatype=="csv":
        fake=pd.read_csv(file_path)
        #chose english rows
        fake=fake[fake['language'].isin(['english'])]
        #delete short title and text
        fake=fake[ fake['title'].str.len() >5]
        fake=fake[ fake['text'].str.len() >10]
        #delete redundant text
        fake.drop_duplicates('title','first',inplace=True)
        fake.drop_duplicates('text','first',inplace=True)

        #choose all the available rows
        mask = [isinstance(item, (str, bytes)) for item in fake['title']]
        fake=fake.loc[mask]
        mask = [isinstance(item, (str, bytes)) for item in fake['text']]
        fake=fake.loc[mask]

        #align all the words to capitals
        # fake['title']=fake['title'].str.upper()
        fake['text']=fake['text'].str.lower()

        df=pd.DataFrame(fake['text'].apply(sent_tokenize).rename('SENTENCES'))
        # df=pd.DataFrame(fake['title'].apply(word_tokenize_no_punct).rename('WORDS'))
        # df=pd.DataFrame(fake['title'].apply(word_tokenize).rename('WORDS'))
        # tokenizer = RegexpTokenizer(r'\w+')
        # df=pd.DataFrame(fake['title'].apply(tokenizer.tokenize).rename('WORDS'))
        # Iterate over each row 
        for index, rows in df.iterrows(): 
            Row_list.append(rows.WORDS) 
    elif datatype=="ptb":
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                Row_list.append(word_tokenize_no_punct(line))
            f.close()
    
    return Row_list

def pad_data(terms_list , max_len , pad_pre = False):
    if max_len is None:
        max_len = 0
        for terms in terms_list:
            if len(terms) > max_len:
                max_len = len(terms)
    new_terms_list = []
    for terms in terms_list:
        pad_len = max_len-len(terms)
        if pad_len > 0:
            if pad_pre:
                new_terms = [PAD_ID]*pad_len + terms
            else:
                new_terms = terms + [PAD_ID]*pad_len
        else:
            new_terms = terms[-max_len:]
        new_terms_list.append(new_terms)
    return new_terms_list
def load_vocab( vocabulary_path):
    vocab_dict = {}
    vocab_res = {}
    vid = 0
    with open(vocabulary_path, mode="r") as vocab_file: #, encoding='utf-8'
        for w in vocab_file:
            vocab_dict[w.strip()] = vid
            vocab_res[vid] = w.strip()
            vid += 1
    return vocab_dict, vocab_res

class DataConverter(object):
    def __init__(self):
        self.vocab = dict( )

    def build_vocab(self, lines):
        counter = 0
        for line in lines:
            counter += 1
            # if counter % 100000 == 0:
                # print("processing line %d" % counter)
            for w in line:
                if w in [' ','','\t','\n','\r']:#,'[',']','(',')','@','#',',',':','\\','`','\"','–','-'
                    continue
                if w in self.vocab:
                    self.vocab[w] += 1
                else:
                    self.vocab[w] = 1

    def save_vocab(self, vocabulary_path):
        vocab_list = _START_VOCAB+sorted(self.vocab, key=self.vocab.get, reverse=True)
        # if len(vocab_list) > max_vocabulary_size:
        #     vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w") as vocab_file: #, encoding='utf-8'
            for w in vocab_list:
                vocab_file.write(w + "\n")
        # print('save vocab done.')

    def convert(self, data_dir, files, fileout='./save/tokenized_data.txt',datatype='csv'):
        for file in files:
            self.build_vocab(get_sent_list(data_dir+file,datatype=datatype))
        vocab_path = './save/vocab.txt'
        self.save_vocab( vocab_path )
        converted = []
        self.vocab_dict, self.vocab_res = load_vocab( vocab_path )
        # print('start padded.')
        max_len = 0
        for file in files:
            print('file : {}'.format( file ) )
            lines=get_sent_list(data_dir+file,datatype=datatype)
            for line in lines:
                words=line
                word_ids = []
                if len(words) > max_len:
                    max_len = len(words)
                word_ids.append( GO_ID )
                for w in words:
                    if w in self.vocab_dict:
                        word_ids.append(self.vocab_dict[w])
                    else:
                        word_ids.append(UNK_ID)
                word_ids.append( EOS_ID )
                converted.append( word_ids)

        # print('max len: {}'.format( max_len ) )
        # max_len = 50
        # padded = pad_data( converted, max_len+2 )
        padded = pad_data( converted,None)
        # print('padded done.')
        padded = np.array( padded, dtype='int32' )
        # padded should be 2D 
        # print(padded.shape)
        np.savetxt(fileout,padded,fmt="%d")
        # print('done.')

def load_data( fn = './save/tokenized_data.txt' ):
    # data = pickle.load( open(fn,'rb' ) )
    data=np.loadtxt(fn)
    return data

if __name__ == '__main__':
    data_dir = './save/'
    files = ['ptb.train.txt']#'fake.csv'
    converter = DataConverter( )
    converter.convert( data_dir,files,datatype='ptb')
