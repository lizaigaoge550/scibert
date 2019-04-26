import torch.nn as nn
from allennlp.modules.token_embedders import BertEmbedder
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.seq2vec_encoders import CnnEncoder
import torch.nn.functional as F
from data import Vocab
from pytorch_pretrained_bert.tokenization import WordpieceTokenizer
from allennlp.data.dataset_readers import text_classification_json
from functools import reduce
import operator

#bert, token_character -->(concat) --> lstm --> crf
class PretrainedBertEmbedder(BertEmbedder):
    def __init__(self, pretrained_model, requires_grad=False, top_layer_only=False):
        model = BertModel.from_pretrained(pretrained_model)
        for param in model.parameters():
            param.requires_grad = requires_grad
        super().__init__(bert_model=model, top_layer_only=top_layer_only)


class TokenCharactersEncoder(nn.Module):
    def __init__(self, embedding, encoder, dropout=0.5):
        super().__init__()
        self._embedding = embedding
        self._encoder = encoder
        if self.training:
            self._dropout = nn.Dropout(p=dropout)
        else:
            self._dropout = nn.Dropout(p=0.0)

    def get_output_dim(self):
        return self._encoder._module.get_output_dim()

    def forward(self, token_characters):
        mask = (token_characters != 0).long()
        return self._dropout(self._encoder(self._embedding(token_characters), mask))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_size, n_layers=2, bidirectional=True):
        super().__init__()
        self.bi_lstm = nn.LSTM(input_dim. hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(2*hidden_dim,tag_size)

    def forward(self, input, mask):
        lstm_output, _ = self.bi_lstm(input) #batch, seq_len, 2*hid_dim
        lstm_output *= mask.unsqueeze(-1)
        encoder = self.linear(lstm_output)
        return encoder * mask.unsqueeze(-1) #batch, seq_len, tag_size



if __name__ == '__main__':
    bert_vocab = Vocab('../scibert-master/scibert_scivocab_uncased/vocab.txt')
    tokernizer = WordpieceTokenizer(bert_vocab.word_to_idx)

    sentence = "The problem is formulated in a manner that subsumes structure from motion"
    words = list(reduce(operator.add, [tokernizer.tokenize(word) for word in sentence.split()]))
    print(words)


    vocab = Vocab('../scibert-master/vocab.txt')

    cnnEncoder = CnnEncoder(embedding_dim=16, num_filters=128, ngram_filter_sizes=[3], conv_layer_activation=nn.ReLU())
    embedding = nn.Embedding(16, len(vocab))
    bert_model = PretrainedBertEmbedder("../scibert-master/scibert_scivocab_uncased/weights.tar.gz")
    tokencharacter_encoder = TokenCharactersEncoder(cnnEncoder,embedding)

    print('model load finished......')










