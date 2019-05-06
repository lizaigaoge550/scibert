import torch
import numpy as np
import torch.nn as nn

def get_final_encoder_states(encoder_outputs, mask, bidirectional=False):
    last_word_indices = mask.sum(1).long - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2)]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_mask(lens):
    '''
    :param lens: list of batch, every item is a int means the length of a sample
    :return: [batch, max_seq_len]
    '''
    max_len = max(lens)
    batch_size = len(lens)
    seq_range = torch.arange(max_len).long()
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)

    seq_length = lens.unsqueeze(1).expand(batch_size, max_len)
    mask = seq_range < seq_length
    return mask.float()





def get_mask_2(lens):
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.FloatTensor(batch_size, max_len)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l]._fill(1.0)
    return mask



def get_char_mask(lens):
    '''
    :param lens: list of batch, in particularly, every item is a list, and every item in the list means the length of word
    :return: mask [batch, max_seq_len, max_word_len]
    '''
    max_seq_len = len(max(lens, key=len))
    tensor_len = torch.zeros((len(lens), max_seq_len))

    #first trunk every len to max_seq_len
    for i in range(len(lens)):
        for j in range(len(lens[i])):
            tensor_len[i,j] = lens[i][j]

    batch_size, seq_len = tensor_len.size()
    max_word_len = torch.max(tensor_len).int().item()
    seq_range = torch.arange(max_word_len).long()
    seq_range = seq_range.view(1,1,max_word_len).expand(batch_size, seq_len, max_word_len)

    seq_length = lens.unsqueeze(-1).expand(batch_size, seq_len, max_word_len)
    mask = seq_range < seq_length
    return mask.float()



def lstm_encoder(sequence, lstm, seq_lens, init_states, is_mask=False, get_final_output=False):
    batch_size = sequence.size(0)
    assert len(seq_lens) == batch_size
    sort_ind = np.argsort(seq_lens)[::-1].tolist()
    sort_seq_lens = [seq_lens[i] for i in sort_ind]
    emb_sequence = reorder_sequence(sequence, sort_ind)

    init_states = (init_states[0].contiguous(), init_states[1].contiguous())

    packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, sort_seq_lens)
    packed_out, final_states = lstm(packed_seq, init_states)
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
    back_map = {ind : i for i, ind in enumerate(sort_ind)}
    reorder_ind = [back_map[i] for i in range(len(sort_ind))]
    lstm_out = reorder_sequence(lstm_out, reorder_ind)
    final_states = reorder_lstm_states(final_states, reorder_ind)

    if is_mask:
        mask = get_mask(seq_lens) # batch, max_seq_lens
        assert lstm_out.size(1) == mask.size(1)
        lstm_out *= mask.unsqueeze(-1)
        return lstm_out, final_states #[batch, max_seq_lens, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])

    if get_final_output:
        mask = get_mask(seq_lens)
        lstm_out = get_final_encoder_states(lstm_out, mask, bidirectional=True)
        return lstm_out, final_states #[batch, hid_dim], ([n_layer, batch, hid_dim], [n_layer, batch, hid_dim])


def reorder_sequence(emb_sequence, order):
    order = torch.LongTensor(order)
    return emb_sequence.index_select(index=order, dim=0)



def reorder_lstm_states(states, order):
    assert isinstance(states, tuple)
    assert len(states) == 2
    assert states[0].size() == states[1].size()
    assert len(order) == states[0].size()[1]

    order = torch.LongTensor(order)
    sorted_states = (states[0].index_select(index=order, dim=1), states[1].index_select(index=order, dim=1))
    return sorted_states






