class Vocab():
    def __init__(self, vocab_file):
        self.word_to_idx = {}
        self.id_to_word = {}
        with open(vocab_file, 'r') as fr:
            for line in fr.readlines():
                word = line.split('\n')[0]
                self.word_to_idx[word] = len(self.word_to_idx)
                self.id_to_word[len(self.word_to_idx)] = word

    def word2id(self, word):
        return self.word_to_idx.get(word, self.word_to_idx['@@UNKNOWN@@'])

    def id2word(self, id):
        return self.id_to_word[id]

    def __len__(self):
        return len(self.word_to_idx)


