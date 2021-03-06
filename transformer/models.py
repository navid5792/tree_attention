from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import data_utils

from transformer.modules import Linear
from transformer.modules import PosEncoding
from transformer.layers import EncoderLayer, DecoderLayer, \
                               WeightedEncoderLayer, WeightedDecoderLayer


def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    pad_attn_mask = pad_attn_mask.expand(b_size, len_q, len_k) # b_size x len_q x len_k

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        #print(subsequent_mask)
        #asas
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, src_vocab_size, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        #self.embedding_table = nn.Embedding(src_vocab_size, d_model, padding_idx=data_utils.PAD,)
#        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        
        self.pos_emb= nn.Embedding(337, 300, padding_idx=0)
        self.pos_emb.weight.data = position_encoding_init(337, 300)
        
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, mask_a):
        #enc_outputs = self.embedding_table(enc_inputs)
        return_attn=False
#        print(enc_inputs.size(), enc_inputs_len.size(), mask_a.size())
#        asas
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len) # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(mask_a, mask_a) # enc_inputs, enc_inputs

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
#        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD,)
#        self.pos_emb = PosEncoding(25, 300) # TODO: *10 fix
        
        self.pos_emb= nn.Embedding(40, 300, padding_idx=0)
        self.pos_emb.weight.data = position_encoding_init(40, 300)
        
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, mask_enc, enc_outputs, mask_dec):
#        dec_outputs = self.tgt_emb(dec_inputs)
        return_attn = False
        dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_dec) # dec_inputs, dec_inputs
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(mask_dec)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_enc)
        
#        print(dec_self_attn_pad_mask.size(), dec_self_attn_subsequent_mask.size(), dec_self_attn_mask.size(), dec_enc_attn_pad_mask.size())
#        print(dec_enc_attn_pad_mask.size())

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, self_attn_mask=dec_self_attn_mask, enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns

class Encoder_Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(Encoder_Decoder, self).__init__()
        self.d_model = d_model
#        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD,)
#        self.pos_emb = PosEncoding(25, 300) # TODO: *10 fix
        
        self.pos_emb= nn.Embedding(85, 300, padding_idx=0)
        self.pos_emb.weight.data = position_encoding_init(85, 300)
        
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, mask_dec, enc_inputs, enc_inputs_len, mask_enc):
#        dec_outputs = self.tgt_emb(dec_inputs)
        
        dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)
        
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len) # Adding positional encoding # TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_pad_mask = get_attn_pad_mask(mask_enc, mask_enc) # dec_inputs, dec_inputs
        
        dec_self_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_dec) 

        dec_enc_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_enc)
        
        enc_dec_attn_pad_mask = get_attn_pad_mask(mask_enc, mask_dec)

        for layer in self.layers:
            enc_outputs, dec_outputs = layer(dec_outputs, enc_outputs, enc_self_attn_mask=enc_self_attn_pad_mask, dec_self_attn_mask=dec_self_attn_pad_mask, dec_enc_attn_mask=dec_enc_attn_pad_mask, enc_dec_attn_mask=enc_dec_attn_pad_mask)
        return enc_outputs, dec_outputs


#class Encoder_Decoder(nn.Module):
#    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
#                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
#        super(Encoder_Decoder, self).__init__()
#        self.d_model = d_model
##        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD,)
##        self.pos_emb = PosEncoding(25, 300) # TODO: *10 fix
#        
#        self.pos_emb= nn.Embedding(337, 300, padding_idx=0)
#        self.pos_emb.weight.data = position_encoding_init(337, 300)
#        
#        self.dropout_emb = nn.Dropout(dropout)
#        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
#        self.layers = nn.ModuleList(
#            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
#
#    def forward(self, dec_inputs, dec_inputs_len, mask_dec, enc_inputs, enc_inputs_len, mask_enc):
##        dec_outputs = self.tgt_emb(dec_inputs)
#        return_attn = False
#        
#        dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
#        dec_outputs = self.dropout_emb(dec_outputs)
#        
#        enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len) # Adding positional encoding # TODO: note
#        enc_outputs = self.dropout_emb(enc_outputs)
#
#        enc_self_attn_pad_mask = get_attn_pad_mask(mask_enc, mask_enc) # dec_inputs, dec_inputs
#        
#        dec_self_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_dec) 
#
#        dec_enc_attn_pad_mask = get_attn_pad_mask(mask_dec, mask_enc)
#        
#
#        dec_self_attns, dec_enc_attns = [], []
#        for layer in self.layers:
#            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, enc_self_attn_mask=enc_self_attn_pad_mask, dec_self_attn_mask=dec_self_attn_pad_mask, enc_dec_attn_mask=dec_enc_attn_pad_mask)
#            if return_attn:
#                dec_self_attns.append(dec_self_attn)
#                dec_enc_attns.append(dec_enc_attn)
#
#        return dec_outputs, dec_self_attns, dec_enc_attns

class tree_encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, tgt_vocab_size, dropout=0.1, weighted=False):
        super(tree_encoder, self).__init__()
        self.d_model = d_model
#        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD,)
#        self.pos_emb = PosEncoding(25, 300) # TODO: *10 fix

        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, parent_inputs, mask_parent, child_inputs, mask_child):

        parent_self_attn_pad_mask = get_attn_pad_mask(mask_parent, mask_parent) # dec_inputs, dec_inputs
        
        child_self_attn_pad_mask = get_attn_pad_mask(mask_child, mask_child) 

        child_parent_attn_pad_mask = get_attn_pad_mask(mask_child, mask_parent)
        
        parent_child_attn_pad_mask = get_attn_pad_mask(mask_parent, mask_child)

        for layer in self.layers:
            outputs = layer(parent_inputs, child_inputs, parent_self_attn_mask = parent_self_attn_pad_mask, child_self_attn_mask=child_self_attn_pad_mask, child_parent_attn_mask=child_parent_attn_pad_mask, parent_child_attn_mask=parent_child_attn_pad_mask)
        return outputs


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
#        self.encoder = Encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
#                               opt.max_src_seq_len, opt.src_vocab_size, opt.dropout, opt.weighted_model)
#        self.decoder = Decoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
#                               opt.max_tgt_seq_len, opt.tgt_vocab_size, opt.dropout, opt.weighted_model)
        self.encoder = tree_encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                       opt.max_tgt_seq_len, opt.tgt_vocab_size, opt.dropout, opt.weighted_model)

    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)

        return param_groups
    
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
#        enc_freezed_param_ids = set(map(id, self.encoder.pos_emb.parameters()))
#        enc_embeddings = set(map(id, self.encoder.embedding_table.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.pos_emb.parameters()))
#        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        freezed_param_ids = dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def tree_encode(self, x, h):
        parent_inputs = x.unsqueeze(0)
        child_inputs = h
        mask_parent = torch.tensor([1]).unsqueeze(0).cuda() 
        if child_inputs.size(0) == 1:
            mask_child = torch.ones(child_inputs.size(0)).expand_as(mask_parent).cuda()
        else:
            mask_child = torch.ones(child_inputs.size(0)).unsqueeze(0).cuda()
        
        child_inputs = child_inputs.transpose(0,1)
#        print(mask_parent.size(), "--", mask_child.size(), parent_inputs.size(), child_inputs.size())

        
        return self.encoder(parent_inputs, mask_parent, child_inputs, mask_child)

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass