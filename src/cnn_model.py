from torch import nn
import torch

from utils import load_embeddings

GLOVE_PATH = '../embeddings/glove.840B.300d/glove.840B.300d.txt'
WIKI_PATH = '../embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
GOOGLE_NEWS_PATH = '../embeddings/GoogleNews-vetors-negative300/GoogleNews-vetors-negative300.bin'
PARAGRAM_PATH = '../embeddings/paragram_300_sl999/paragram_300_sl999.txt'

class EmbeddingLayer(nn.Module):
    
    def __init__(self, word_count):
        super().__init__()
        
        self.word_count = word_count
        self.num_words = len(word_count)
        
        embedding_weights = self.load_all_embeddings(word_count, self.num_words)
            
        self.original_embedding_weights = embedding_weights
        self.embeddings = nn.Embedding(num_words + 1, embedding_size, padding_idx=0)
        self.embeddings.weight = nn.Parameter(embedding_weights)
        self.embeddings.weight.requires_grad = False
        
    def forward(self, x):
        embedding = self.embeddings(x)
        if self.training:
            embedding += torch.randn_like(embedding)
        return embedding
    
    def reset_weights(self):
        self.embeddings.weight = nn.Parameter(self.original_embedding_weights)
        self.embeddings.weight.requires_grad = False
    
    def load_all_embeddings(self, word_count, num_words):
        glove_embeddings = load_embeddings(GLOVE_PATH, word_count, num_words)
        wiki_embeddings = load_embeddings(WIKI_PATH, word_count, num_words)
        GOOGLE_NEWS_PATH = load_embeddings(GOOGLE_NEWS_PATH, word_count, num_words)
        paragram_embeddings = load_embeddings(PARAGRAM_PATH, word_count, num_words)

        embedding_matrix = np.concatenate((glove_embeddings,
                                           wiki_embeddings,
                                           GOOGLE_NEWS_PATH,
                                           paragram_embeddings), axis=1)

        return torch.tensor(embedding_matrix, dtype=torch.float32)


class GRULayer(nn.Module):
    # https://blog.floydhub.com/gru-with-pytorch/
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        
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
        return self.gru(x)


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
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
        return lstm_outputs, lstm_states


class CNNModel(nn.Module):
    def __init__(self, embedding_size, hidden_size_1, num_layers_1, hidden_size_2, 
                num_layers_2, dense_size_1, dense_size_2, output_size, embeddings):
        super().__init__()
        
        self.embeddings = embeddings
        self.lstm = LSTMLayer(input_size=embedding_size,
                              hidden_size=hidden_size_1,
                              num_layers=num_layers_1)
        self.gru = GRULayer(input_size=self.hidden_size_1*2,
                            hidden_size=self.hidden_size_2,
                            num_layers=self.num_layers_2)
        
        self.linear = nn.Linear(dense_size_1, dense_size_2)
        self.batch_norm = torch.nn.BatchNorm1d(dense_size_2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(dense_size_1, output_size)
        
    def forward(self, x, features):
        h_embedding = self.embeddings(x)
        o_lstm, h_lstm = self.lstm(h_embedding)
        o_gru, h_gru = self.gru(o_lstm)
        
        avg_pool = torch.mean(o_gru, 1)
        max_pool, _ = torch.max(o_gru, 1)

        concat = torch.cat([h_lstm, h_gru, avg_pool, max_pool, features], 1)
        concat = self.linear(concat)
        concat = self.batch_norm(concat)
        concat = self.relu(concat)
        out = self.out(concat)
        
        return out