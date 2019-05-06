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
        init.orthogonal_(self._init_h)
        init.orthogonal_(self._init_c)
        init.uniform_(self.embedding.weight, -0.01, 0.01)
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        if char_vocab_size:
            self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size)
            self.char_encoder = CnnEncoder(embedding_dim=char_embedding_size, num_filters=num_filter, ngram_filter_sizes=ngram_filter_size)

    #sequence, lstm, seq_lens, init_states, is_mask=False, get_final_output=False:
    def _lstm_forward(self, emb_input, input_lens):
        #emb_input [batch, seq_len, emb_dim]
        batch_size = emb_input.size(0)
        encoder_output, encoder_states = lstm_encoder(emb_input, self.lstm, input_lens, (self._init_h.unsqueeze(1).repeat(1,batch_size,1),
                                                                                         self._init_c.unsqueeze(1).repeat(1,batch_size,1)), is_mask=True)
        return encoder_output, encoder_states

    def cnn_forward(self, char_emb_input, char_input_lens):
        #char_emb_input [batch, seq_len, word_len, char_embeding_dim]
        mask = get_char_mask(char_input_lens)  #batch, max_seq_len, max_word_len
        batch_size, max_seq_len, max_word_len = mask.size()
        assert char_emb_input.size(0) == batch_size
        for i in range(batch_size):
            char_emb_input[i] = char_emb_input[i,:max_seq_len,:max_word_len,:]
        assert char_emb_input.size()[1] == max_seq_len
        assert char_emb_input.size()[2] == max_word_len
        char_emb_input = char_emb_input.view(batch_size*max_seq_len, max_word_len, -1)
        mask = mask.view(batch_size*max_seq_len, -1)
        char_emb_input = self.char_encoder(char_emb_input, mask)
        char_emb_input = char_emb_input.view(batch_size, max_seq_len, -1)
        return char_emb_input


    def forward(self, **kwargs):
        word_sequence = kwargs['word_sequence']
        char_sequence = kwargs['char_sequence']
        word_sequence_len = kwargs['word_sequence_lens']
        char_sequence_len = kwargs['char_sequence_lens']

        word_seq_emb = self.embedding(word_sequence)
        char_seq_emb = self.char_embedding(char_sequence)

        encoder_output, _ = self._lstm_forward(word_seq_emb, word_sequence_len)
        char_encoder_output = self.cnn_forward(char_seq_emb, char_sequence_len)
        print(f'encoder output size : {encoder_output.size()}')
        print(f'char_encoder output size : {char_encoder_output.size()}')
if __name__ == '__main__':
    word_sequence = torch.LongTensor([[1,2,3,4,5], [1,2,3,4,0]])
    char_sequence = torch.LongTensor([[[1,1,1,1,1], [2,2,2,0,0], [3,3,3,0,0], [4,4,4,4,0], [5,5,0,0,0]],
                                  [[1,1,1,1,1], [2,2,2,0,0], [3,3,3,0,0], [4,4,4,4,0], [0,0,0,0,0]]
                                  ])
    word_sequence_lens = torch.LongTensor([5,4])
    char_sequence_lens = torch.LongTensor([[5,3,3,4,2],
                          [5,3,3,4,0]])

    model = BaseModel(vocab_size=10, input_dim=10, output_dim=10, char_vocab_size=10, char_embedding_size=5, num_filter=10, ngram_filter_size=[2])
    model(word_sequence = word_sequence, char_sequence = char_sequence, word_sequence_lens = word_sequence_lens, char_sequence_lens=char_sequence_lens)

