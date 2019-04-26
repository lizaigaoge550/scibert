import torch
from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer

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

