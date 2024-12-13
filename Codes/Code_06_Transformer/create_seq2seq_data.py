import pandas as pd
from torch.utils.data import Dataset
from itertools import chain

# Generates Data for SequenceToSequence Models.
# Input:
#     a. source_data_file
#     b. target_data_file
#     c. extra_token_config: { "unkonwn":"<unk>", "pad":"<pad>"", "eos":"<eos>", "sos":"<sos>"}
#     d. max_sequence_length
# Output (X denoted input, Y denotes output):
#     a. X_encoder is the matrix of token-indices in Source sentences, each sentence suffixed by index of "<eos>" token. 
#     b. X_decoder is the matrix of token-indices in Target sentences, each sentence prefixed by index of "<sos>" token. X_decoder is required because we want to do Teacher Forcing, 
#         which means that we want to provide the correct current token to decoder to predict the next token, instead of relying only on its own prediction.
#     c. Y is the matrix token-indices in Target sentences, each sentence suffixed by index of "<eos>" token.



'''
Preprocess Data
1. Lowercasing, removing stopwords. <br>
2. Stemming, Lemmetization. <br>
3. Tokenization. <br>
4. Here, we are just doing tokenization by splitting on space.
'''
class Preprocessor:
    def __init__(self):
        self.tokenize_on = " "

    def lower_case(self, text_string):
        '''
        text_string = "This is One sentence."
        returns "this is one sentence."
        '''
        return text_string.lower()
    
    def handle_punctuations(self, text_string):
        '''
        text_string = "this is, one sentence."
        returns "this is , one sentence ."
        '''
        text_string = text_string.replace(',',' ,').replace('.',' .')
        return text_string
    
    def tokenize(self,text_string):
        '''
        text_string = "This is one sentence."
        returns token_list = ["This","is","one","sentence."]
        '''
        token_list = text_string.split(self.tokenize_on)
        return token_list
    
    def transform(self, text_string):
        text_string = self.lower_case(text_string)
        text_string = self.handle_punctuations(text_string)
        token_list = self.tokenize(text_string)
        return token_list

'''Build Vocab
1. Generally, in a SequenceToSequence Problem, Vocab is created from both the Source and the Target Tokens, consider Sentence Translation for ex, where Source and Target tokens can be in different languages.
2. We can create Vocab separately for Source and Target tokens, or can create shared vocab. Shared Vocab is preferable though.
3. Also, Vocab is generated from only Training Data, but are using full Dataset to create Vocab, as our datasize is small.
'''
class VocabBuilder:
    def __init__(self,token_corpus, extra_tokens_to_indices):
        '''
        token_corpus = ['tools', 'and', 'a', 'man', 'gardening', 'inside', 'two', 'holding', 'are', '.']
        '''
        self.token_corpus = token_corpus
        self.extra_tokens_to_indices = extra_tokens_to_indices
                        
    def get_vocabs(self):
        token_to_index = dict(self.extra_tokens_to_indices)
        all_unique_tokens = set(self.token_corpus).difference(set(list(token_to_index.keys()))) # remove all those tokens in text_corpus which are already present in extra_tokens_to_indices
        for index, token in enumerate(all_unique_tokens):
            token_to_index[token] = index + len(self.extra_tokens_to_indices)
        index_to_token = {v:k for k,v in token_to_index.items()}
        return token_to_index, index_to_token
'''
Maps Token To Vocab-Indices and Vice-Versa based on Vocabulary
'''
class TokenIndexMapper:
    def __init__(self):
        pass
    
    def get_encoding(self,sentence, token_to_index, unknown_token):
        '''
        sentence must be a list of tokens.
        Ex: ["Climate","change","is","a","pressing","global","issue"]
        '''
        encoded_sentence = []
        for token in sentence:
            if token in token_to_index: encoded_sentence.append(token_to_index[token])
            else: encoded_sentence.append(token_to_index[unknown_token])
        return encoded_sentence
    
    def get_decoding(self,encoded_sentence, index_to_token):
        '''
        encoded_sentence must be a list of vocab indices.
        Ex: encoded_sentence = [24,21,4,1,..] 
        '''
        sentence = [index_to_token[index] for index in encoded_sentence]
        return " ".join(sentence)

class Sequence2SequenceData(Dataset):
    def __init__(self, source_data_file, target_data_file, extra_token_config, max_sequence_length):
        self.source_file = source_data_file
        self.target_file = target_data_file
        self.extra_token_config = extra_token_config
        self.max_sequence_length = max_sequence_length
        self.X_encoder_tokens, self.X_decoder_tokens, self.Y_tokens = self.create_tokenized_data()
        self.X_encoder_indices = self.map_tokens_to_indices(max_sequence_length, self.X_encoder_tokens)
        self.X_decoder_indices = self.map_tokens_to_indices(max_sequence_length, self.X_decoder_tokens)
        self.Y_indices = self.map_tokens_to_indices(max_sequence_length, self.Y_tokens)
    
    def read_data(self):
        self.source_df = pd.read_csv(self.source_file,sep="\t",header=None)
        self.target_df = pd.read_csv(self.target_file,sep="\t",header=None)
        self.source_df.columns = ["Source"]
        self.target_df.columns = ["Target"]
        self.full_df = pd.concat([self.source_df,self.target_df],axis=1)

    def pre_process_data(self):
        preprocessor = Preprocessor()
        self.source_df["Source"] = self.source_df["Source"].apply(lambda x: preprocessor.transform(x))
        self.target_df["Target"] = self.target_df["Target"].apply(lambda x: preprocessor.transform(x))
    
    def build_vocab(self):
        token_corpus_source = list(chain.from_iterable(self.source_df["Source"].tolist())) # flattens a 2D list to 1D
        token_corpus_target = list(chain.from_iterable(self.target_df["Target"].tolist())) # flattens a 2D list to 1D
        token_corpus = list(set(token_corpus_source + token_corpus_target))
        extra_tokens_to_indices = {}
        if "unknown" in self.extra_token_config: self.unknown_token = self.extra_token_config["unknown"]
        if "pad" in self.extra_token_config: self.pad_token = self.extra_token_config["pad"]
        if "eos" in self.extra_token_config: self.eos_token = self.extra_token_config["eos"]
        if "sos" in self.extra_token_config: self.sos_token = self.extra_token_config["sos"]
        extra_tokens_to_indices[self.unknown_token] = 0
        extra_tokens_to_indices[self.pad_token] = 1
        extra_tokens_to_indices[self.eos_token] = 2
        extra_tokens_to_indices[self.sos_token] = 3

        vocab_builder = VocabBuilder(token_corpus, extra_tokens_to_indices)
        self.token_to_index, self.index_to_token = vocab_builder.get_vocabs()

    def create_tokenized_data(self):
        self.read_data()
        self.pre_process_data()
        self.build_vocab()

        source_sentences = self.source_df["Source"].tolist()
        target_sentences = self.target_df["Target"].tolist()
        X_encoder_tokens = [el + [self.eos_token] for el in source_sentences]
        X_decoder_tokens = [[self.sos_token] + el for el in target_sentences]
        Y_tokens = [el + [self.eos_token] for el in target_sentences]
        return X_encoder_tokens, X_decoder_tokens, Y_tokens

    def map_tokens_to_indices(self, max_sequence_length, token_matrix):
        token_index_mapper = TokenIndexMapper()
        index_matrix = []
        for el in token_matrix:
            el = el[:max_sequence_length] # truncate sentence to max_seq_length
            if len(el) < max_sequence_length: # pad sentence to max_seq_length
                pad_tokens_to_append = max_sequence_length - len(el)
                el = el + [self.pad_token]*pad_tokens_to_append
            index_matrix.append(token_index_mapper.get_encoding(el, self.token_to_index, self.unknown_token))
        return index_matrix

    def __len__(self):
        return len(self.X_encoder_indices)

    def __getitem__(self, idx):
        X_encoder_token_indices = self.X_encoder_indices[:,idx]
        X_decoder_token_indices = self.X_decoder_indices[:,idx]
        Y_token_indices = self.Y_indices[:,idx]
        return X_encoder_token_indices, X_decoder_token_indices, Y_token_indices
