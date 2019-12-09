from torch import nn
import torch
import numpy as np

from utils import load_embeddings

GLOVE_PATH = '../../embeddings/glove.840B.300d/glove.840B.300d.txt'
WIKI_PATH = '../../embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
GOOGLE_NEWS_PATH = '../../embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
PARAGRAM_PATH = '../../embeddings/paragram_300_sl999/paragram_300_sl999.txt'

class EmbeddingLayer(nn.Module):
    
    def __init__(self, word_index, word_count):
        super().__init__()
        
        self.word_index = word_index
        self.num_words = word_count
        
        embedding_weights = self.load_all_embeddings(word_index, self.num_words)
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float32)
            
        self.original_embedding_weights = embedding_weights
        print(f'num of words: {self.num_words}')
        self.embeddings = nn.Embedding(self.num_words, 300, padding_idx=0)
        self.embeddings.weight = nn.Parameter(embedding_weights)
        self.embeddings.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)

    def dropout1d(self, embedding):
        x = embedding.permute(0, 2, 1)   
        x = self.embedding_dropout(x)
        return x.permute(0, 2, 1)
        
    def forward(self, x):
        embedding = self.embeddings(x)
        if self.training:
            embedding += torch.randn_like(embedding)
        return self.dropout1d(embedding)
    
    def reset_weights(self):
        self.embeddings.weight = nn.Parameter(self.original_embedding_weights)
        self.embeddings.weight.requires_grad = False
    
    def load_all_embeddings(self, word_index, num_words):
        # Word cover rate in the embedding is: 0.8724167059563099
        glove_embeddings = load_embeddings(GLOVE_PATH, word_index, num_words)
        # Word cover rate in the embedding is: 0.6717114568599717
        # wiki_embeddings = load_embeddings(WIKI_PATH, word_index, num_words)
        # google_new_embeddings = load_embeddings(GOOGLE_NEWS_PATH, word_index, num_words)
        # paragram_embeddings = load_embeddings(PARAGRAM_PATH, word_index, num_words)

        embedding_matrix = np.concatenate((glove_embeddings,
                                           # wiki_embeddings,
                                           # google_new_embeddings,
                                           # paragram_embeddings,
                                           ), axis=1)

        return torch.tensor(glove_embeddings, dtype=torch.float32)


class GRULayer(nn.Module):
    # https://blog.floydhub.com/gru-with-pytorch/
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.init_hidden()
        
    def init_hidden(self):
        # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        output, state = self.gru(x)
        return self.dropout(output), state


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.init_hidden()
        
    def init_hidden(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        lstm_outputs, (lstm_states, _) = self.lstm(x)
        return self.dropout(lstm_outputs), lstm_states


class CNNModel(nn.Module):
    def __init__(self, embedding_size, hidden_size_1, num_layers_1, hidden_size_2, 
                num_layers_2, dense_size_1, dense_size_2, output_size, embeddings, dropout):
        super().__init__()
        
        self.embeddings = embeddings
        self.lstm = LSTMLayer(input_size=embedding_size,
                              hidden_size=hidden_size_1,
                              num_layers=num_layers_1,
                              dropout=dropout)
        self.gru = GRULayer(input_size=hidden_size_1*2,
                            hidden_size=hidden_size_2,
                            num_layers=num_layers_2,
                            dropout=dropout)
        
        self.linear = nn.Linear(dense_size_1, dense_size_2)
        self.batch_norm = torch.nn.BatchNorm1d(dense_size_2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(dense_size_2, output_size)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x, features):
        h_embedding = self.embeddings(x)
        o_lstm, h_lstm = self.lstm(h_embedding)
        o_gru, h_gru = self.gru(o_lstm)
        
        avg_pool = torch.mean(o_gru, 1)
        max_pool, _ = torch.max(o_gru, 1)
        # import pdb; pdb.set_trace()

        h_lstm = torch.cat(h_lstm.split(1, 0), -1).squeeze(0)
        h_gru = torch.cat(h_gru.split(1, 0), -1).squeeze(0)

        concat = torch.cat((h_lstm, h_gru, avg_pool, max_pool, features), 1)
        concat = self.linear(concat)
        concat = self.batch_norm(concat)
        concat = self.relu(concat)
        concat = self.dropout(concat)
        out = self.out(concat)
        
        return out