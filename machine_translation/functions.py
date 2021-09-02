import tensorflow as tf
import re
from unicodedata import normalize
import string
import numpy as np


# clean a list of lines
def clean_preprocess_text(lines):
    clean_pair = list()
    for line in lines:
        # normalize unicode characters
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # convert to lowercase
        line = line.lower()
        # put space before and after punctuation
        line = re.sub(r"([?.!])", r" \1 ", line)
        # replace ' and - with a space
        line = re.sub(r"['-]", r" ", line)
        # remove digits
        remove_digits = str.maketrans('', '', string.digits)
        line = line.translate(remove_digits)
        # reduces spaces into one
        line = re.sub(r" +", r" ", line)
        # remove space at start and end of line
        line = line.strip()
        # add the cleaned line in a new table which will contain all the new sentences
        clean_pair.append(line)

    return np.array(clean_pair)


# fit a tokenizer
def create_tokenizer(lines):
    to_exclude = '"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n'
    to_tokenize = '.!?'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=to_exclude)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    # add 2 to max length because sos_token(start sent) and eos_token(end sent) will be added in the sentence
    return max(len(line.split()) for line in lines) + 2


# encode the corpus and add paddind to max length
def encode_sequences(corpus, tokenizer, max_length, sos_token, eos_token):
    # encode the corpus
    sentences = tokenizer.texts_to_sequences(corpus)

    # add START and END token to the corpus
    sentences = [sos_token + sentence + eos_token for sentence in sentences]

    # Pad the sentences
    sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences,
                                                              value=0,
                                                              padding='post',
                                                              maxlen=max_length
                                                              )
    return sentences
