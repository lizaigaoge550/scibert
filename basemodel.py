import torch
import torch.nn as nn
import torch.nn.init as init
from utils import *
from allennlp.modules.seq2vec_encoders import CnnEncoder

class BaseModel(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim,
                 n_layers=1,
                 dropout=0.0,
                 char_vocab_size = None,
                 char_embedding_size = None,
                 num_filter=None,
                 ngram_filter_size = None
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self._init_h = nn.Parameter(torch.Tensor(n_layers*2, output_dim)) #默认是bidirectional
        self._init_c = nn.Parameter(torch.Tensor(n_layers*2, output_dim))
        init.orthogonal(self._init_h)
        init.orthogonal(self._init_c)
        init.uniform_(self.embedding, -0.01, 0.01)
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        if self.char_vocab_size:
            self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size)
            self.char_encoder = CnnEncoder(char_embedding_size, num_filters=num_filter, ngram_filter_sizes=ngram_filter_size)

    #sequence, lstm, seq_lens, init_states, is_mask=False, get_final_output=False:
    def _lstm_forward(self, emb_input, input_lens):
        #emb_input [batch, seq_len, emb_dim]
        encoder_output, encoder_states = lstm_encoder(emb_input, self.lstm, input_lens, (self._init_h, self._init_c), is_mask=True)


    def cnn_forward(self, char_emb_input, char_input_lens):
        #char_emb_input [batch, seq_len, word_len, char_embeding_dim]
        mask = get_char_mask(char_input_lens)  #batch, max_seq_len, max_word_len
        batch_size, max_seq_len, max_word_len = mask.size()
        assert char_emb_input.size(0) == batch_size
        for i in range(batch_size):
            char_emb_input[i] = char_emb_input[i,:max_seq_len,:max_word_len,:]
        assert char_emb_input.size()[0] == max_seq_len
        assert char_emb_input.size()[1] == max_word_len
        return self.char_encoder(char_emb_input, mask)


    def forward(self, *input):
        pass
