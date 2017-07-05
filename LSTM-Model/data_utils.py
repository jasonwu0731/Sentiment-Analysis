import numpy as np
import re
import os 
import collections
import cPickle
from keras.utils import to_categorical

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def build_data2train(data_folder, vocabulary_size, max_len, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    if not os.path.exists('mr.p') or not os.path.exists('dictionary.p'):
        print("Start loading Data from text ... ")
        # Build Vocab:
        revs = []
        nb_class = 0
        max_len_true = 0
        nb_word = 0
        words_list = []
        for _file in data_folder:
            with open(_file, "rb") as f:
                for line in f:       
                    rev = []
                    rev.append(line.strip()) # returns a copy of the string in which all chars have been stripped from the beginning and the end of the string (default whitespace characters).
                    if clean_string:
                        orig_rev = clean_str(" ".join(rev)) 
                    else:
                        orig_rev = " ".join(rev).lower()
                    words_list += orig_rev.split()
                    if max_len_true < len(orig_rev.split()):
                        max_len_true = len(orig_rev.split())
                    revs.append({'y':nb_class, 'txt':orig_rev})
                    nb_word = nb_word + len(orig_rev.split())
            nb_class += 1
        count = [['UNK', -1]]
        print("Total vocabulary: " + str(len(collections.Counter(words_list))) + ". Use most common "+ str(vocabulary_size) + " only")
        count.extend(collections.Counter(words_list).most_common(vocabulary_size - 1))
        word2idx = dict()
        for word, _ in count:
          word2idx[word] = len(word2idx)
        idx2word = dict(zip(word2idx.values(), word2idx.keys()))

        # Build data from word to idx
        data = []
        label = []
        for sent in revs:
            label.append([sent['y']])
            idx_list = []
            for word in sent['txt'].split():
                if len(idx_list) < 20:
                    if word not in word2idx:
                        idx_list.append(word2idx['UNK'])
                    else:
                        idx_list.append(word2idx[word])
            # Padding
            for i in range(max_len-len(idx_list)):
                idx_list.append(word2idx['UNK'])
            data.append(idx_list)
        data = np.array(data)
        label = np.array(label)
        label = to_categorical(label, num_classes=len(data_folder))
        # shuffle data
        shuffle_idx = np.arange(len(data))
        np.random.shuffle(shuffle_idx)
        data = data[shuffle_idx]
        label = label[shuffle_idx]
        with open('mr.p','wb') as f:
            cPickle.dump([data, label], f)
        with open('dictionary.p','wb') as f:
            cPickle.dump([word2idx, idx2word, max_len_true], f)
    else:
        print("Start loading Data from exist file ... ")
        with open('mr.p','rb') as f:
            [data, label] = cPickle.load(f)
        with open('dictionary.p','rb') as f:
            [word2idx, idx2word, max_len_true] = cPickle.load(f)

    print("Number of sentences: " + str(len(data)))
    print("Vocab size: " + str(len(word2idx)))
    print("Max sentence length: " , max_len_true, ". Restrict max_len to", max_len)
    print("First first sentence idx:", data[0])
    print("First label:", label[0])
    return data, label, word2idx, idx2word

def sent2idx(sent, word2idx, max_len):
    sent = clean_str(sent) 
    idx_list = []
    for word in sent.split():
        if len(idx_list) < 20:
            if word not in word2idx:
                idx_list.append(word2idx['UNK'])
            else:
                idx_list.append(word2idx[word])
    # Padding
    for i in range(max_len-len(idx_list)):
        idx_list.append(word2idx['UNK'])
    return np.array([idx_list])

if __name__ == "__main__":
    data_folder = ["../twitter_neg.csv", "../twitter_pos.csv"]
    revs, vocab = build_data(data_folder)
    print(revs[:10])
