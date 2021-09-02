import tensorflow as tf
import os
import pickle
from machine_translation import model
from machine_translation import functions


###################################### Global parameters  ######################################

model_folder = 'model'


###############################  load the whole model, parameters and variables  #########################

def load_model(DATA_PATH, model_path):
    # Read the parameters model
    print('Loading Model parameters ...')
    model_info_path = DATA_PATH + '/model_info.pkl'
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
    print('Model parameters loaded.\n')

    # sos_token_input = model_info['sos_token_input']
    # eos_token_input = model_info['eos_token_input']
    # sos_token_output = model_info['sos_token_output']
    # eos_token_output = model_info['eos_token_output']
    # input_max_length = model_info['input_max_length']
    # output_max_length = model_info['output_max_length']

    # Create an instance of the Transforer model and load the saved model
    print('Create an instance of the Transforer model and load the saved model ...')
    transformer_model_loaded = model.Transformer(vocab_size_enc=model_info['vocab_size_enc'],
                                                 vocab_size_dec=model_info['vocab_size_dec'],
                                                 d_model=model_info['d_model'],
                                                 n_layers=model_info['n_layers'],
                                                 FFN_units=model_info['ffn_dim'],
                                                 n_heads=model_info['n_heads'],
                                                 dropout_rate=model_info['drop_rate']
                                                 )
    # Load the saved model
    transformer_model_loaded.load_weights(model_path + '/transformer')
    print('Model loaded.\n')

    # Read the training and validation data
    print("loading train data and valid data ...")
    train_valid_data_path = DATA_PATH + '/train_valid_data.pkl'
    with open(train_valid_data_path, 'rb') as f:
        train_valid_data = pickle.load(f)

    encoder_inputs = train_valid_data['encoder_inputs_train']
    decoder_outputs = train_valid_data['decoder_outputs_train']
    encoder_inputs_valid = train_valid_data['encoder_inputs_valid']
    decoder_outputs_valid = train_valid_data['decoder_outputs_valid']
    print("Train data and valid data loaded.\n")

    # Read the parameters tokenizers with the vocabularies
    print('Loading the parameters tokenizers ....')
    input_vocab_file = DATA_PATH + '/tokenizer_inputs.pkl'
    with open(input_vocab_file, 'rb') as f:
        tokenizer_inputs = pickle.load(f)

    output_vocab_file = DATA_PATH + '/tokenizer_outputs.pkl'
    with open(output_vocab_file, 'rb') as f:
        tokenizer_outputs = pickle.load(f)

    print('Tokenizers loaded.\n')

    return transformer_model_loaded, tokenizer_inputs, tokenizer_outputs, model_info


####################################  functions to make prediction  ##############################

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target corresponding to the given sequence
def seq2text(tokenizer, sequences):
    target = list()
    for i in sequences:
        word = word_for_id(i, tokenizer)
        if word is None:
            continue
        target.append(word)

    return ' '.join(target)


# clean the sentence and encode it in sequence
def encode_sentence(input_sentence, tokenizer, max_length, sos_token, eos_token):
    inp_sentence_cleaned = functions.clean_preprocess_text(input_sentence.split('\n'))
    inp_sequence = functions.encode_sequences(inp_sentence_cleaned, tokenizer, max_length, sos_token, eos_token)
    # inp_sequence shape: (1, max_length) => [[sequence]]
    return inp_sequence[0]


# Transform the sequence of tokens to a sentence
def predict_from_seq(inp_sequence, transformer_model_loaded, output_max_length, tokenizer_outputs, sos_token_output, eos_token_output):
    # Reshape the input
    enc_input = tf.expand_dims(inp_sequence, axis=0)
    # Set the initial output sentence to sos
    out_sentence = sos_token_output
    # Reshape the output
    output = tf.expand_dims(out_sentence, axis=0)

    # For max target len tokens
    for _ in range(output_max_length):
        # Call the transformer and get the logits
        predictions = transformer_model_loaded(enc_input, output, False)  # (1, seq_length, VOCAB_SIZE_OUTPUT)
        # Extract the logists of the next word
        prediction = predictions[:, -1:, :]
        # The highest probability is taken
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        # Check if it is the eos token
        if predicted_id == eos_token_output:
            break
        # Concat the predicted word to the output sequence
        output = tf.concat([output, predicted_id], axis=-1)

    output_final = tf.squeeze(output, axis=0).numpy()

    predicted_sentence = seq2text(tokenizer_outputs, [i for i in output_final])

    return predicted_sentence


def predict(inp_sentence, transformer_model_loaded, tokenizer_inputs, tokenizer_outputs, input_max_length, output_max_length, sos_token_output, eos_token_output, sos_token_input, eos_token_input):
    # generate sequence corresponding to the given sentence
    inp_sequence = encode_sentence(inp_sentence, tokenizer_inputs, input_max_length, sos_token_input, eos_token_input)
    # return predicted sentence
    return predict_from_seq(inp_sequence, transformer_model_loaded, output_max_length, tokenizer_outputs, sos_token_output, eos_token_output)


def translate(sentence, lang_selected):
    if lang_selected == "Francais ==> Portugais":
        data_path = 'machine_translation/mini_projet/data_fr_por'
        model_path = os.path.abspath(os.path.join(data_path, model_folder))
        transformer_model_loaded, tokenizer_inputs, tokenizer_outputs, model_info = load_model(data_path, model_path)
    elif lang_selected == "Anglais ==> Francais":
        data_path = 'machine_translation/mini_projet/data_eng_fr'
        model_path = os.path.abspath(os.path.join(data_path, model_folder))
        transformer_model_loaded, tokenizer_inputs, tokenizer_outputs, model_info = load_model(data_path, model_path)
    else:
        return ""

    sos_token_input = model_info['sos_token_input']
    eos_token_input = model_info['eos_token_input']
    sos_token_output = model_info['sos_token_output']
    eos_token_output = model_info['eos_token_output']
    input_max_length = model_info['input_max_length']
    output_max_length = model_info['output_max_length']

    # Get the predicted sentence for the input sentence
    return predict(sentence,
                   transformer_model_loaded,
                   tokenizer_inputs, tokenizer_outputs,
                   input_max_length,
                   output_max_length,
                   sos_token_output,
                   eos_token_output,
                   sos_token_input,
                   eos_token_input
                   )
