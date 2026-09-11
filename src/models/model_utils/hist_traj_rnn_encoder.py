import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None) -> torch.Tensor:
    bs, tf = original_seqs.shape[:2] # B, T(num of the past frame)
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1 # [tf, tf, tf,..., tf] (bs,1)

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices): # i, tf
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len]) # select the (i,0:tf) of original_seqs and pad into the list
    # if the lower indices is not 0, it is also ok
    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) # pack the list of batch, and deal with the case padding with 0 when occlusion (delete those 0)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs) # input (batch sizes * variable seq_length), output (B, variable seq_length, hidden_size)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length) # inverse of pack, (B, T=20, hidden_size)
    # Tuple of Tensor containing the padded sequence, and a Tensor containing the list of lengths of each sequence in the batch. 

    return output, (h_n, c_n) # (B,T,), 

class Encoding(object):
    def __init__(self, model_registrar,
                 device, hist_enc_dim):
        super(Encoding, self).__init__()
        self.device = device

        self.model_registrar = model_registrar
        self.node_modules = dict()
        self.node_modules['node_history_encoder'] = self.model_registrar.get_model('node_history_encoder', 
                                                                                   model_if_absent=nn.LSTM(input_size=8,
                                                                                    hidden_size=hist_enc_dim,
                                                                                    batch_first=True)).to(self.device)
        

    def encode_hist(self, node_hist, dropout_keep_prob, first_history_indices=None) -> torch.Tensor:
        """
        Encodes the nodes history.
        :param batch: Batch of data.
        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        num_agents = node_hist.shape[0] 
        if first_history_indices is None:
            first_history_indices = torch.zeros(num_agents, dtype=torch.int)
        node_hist = node_hist.to(self.device)

        x_concat_list = list()
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules['node_history_encoder'],
                                                        original_seqs=node_hist,
                                                        lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1 - dropout_keep_prob,
                            training=True)  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        
        # node_history_encoded = outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
        node_history_encoded = outputs[:,-1,:]
        # node_history_encoded = self.encode_node_history(True,
        #                                         node_hist,
        #                                         first_history_indices, dropout_keep_prob)
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]
        encoding = torch.cat(x_concat_list, dim=1)
        return encoding
