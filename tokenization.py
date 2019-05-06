from utils import tokens_to_indices
import numpy as np

class TokenizerToIndex():
    def __init__(self, tokenizer, word_vocab, char_vocab, **kwargs):
        self.tokenizer = tokenizer
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.kwargs = kwargs

    def chunk(self, word_idx, offset, max_seq_len):
        while len(offset) > max_seq_len:
            idx = offset.pop()
        return word_idx[:idx], offset

    def char_to_index(self, char_vocab, chars):
        return [char_vocab.word2id(char) for char in chars]

    def __call__(self, x):
        # x可以使用分词器来处理
        x = list(filter(lambda x:x != ' ' and x != '' ,x.split()))
        word_idx, offset = tokens_to_indices(x , vocab=self.word_vocab, wordpiece_tokenizer=self.tokenizer)
        max_word_len = self.kwargs['max_word_len']
        max_seq_len = self.kwargs['max_seq_len']

        if len(offset) < max_seq_len:
            offset += [0] * (max_seq_len - len(offset))
            word_idx += [0] * (max_seq_len - len(offset))
        elif len(offset) > max_seq_len:
            word_idx, offset = self.chunk(word_idx, offset, max_seq_len)

        assert len(offset) == max_seq_len

        char_x = [self.char_to_index(self.char_vocab, word) for word in x]
        char_x_len = [len(word) for word in x]
        char_new_x = np.zeros((max_seq_len, max_word_len)) * 0
        char_new_x_len = []
        for i, l in enumerate(char_x_len):
            if i == max_seq_len:
                break
            if l:
                if l >= max_word_len:
                    char_new_x[i, :] = char_x[i][:max_word_len]
                    char_new_x_len.append(max_word_len)
                elif l < max_word_len:
                    char_new_x[i, :l] = char_x[i]
                    char_new_x_len.append(l)
            else:
                char_new_x_len.append(0)
        char_new_x_len += [0] * (max_seq_len - len(char_new_x_len))
        assert len(char_new_x_len) == max_seq_len