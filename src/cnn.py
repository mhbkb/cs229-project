# System import
import collections

# External import
import torch
import psutil
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Internal import
from data_pre_processing import load_data
from utils import timer
from cnn_model import *

TRAIN_PATH = 'preprocess.csv'
TEST_PATH = 'test.csv'

CUDA = True if torch.cuda.is_available() else False


print(f'CUDA available: {CUDA}')
print(f'Total cpus: {psutil.cpu_count()}')


def get_data():
    return load_data(TRAIN_PATH)


@timer
def build_word_dict(pd_data):
    word_dict = collections.defaultdict(int)

    for setences in tqdm(pd_data['question_text'].values, disable=False):
        for word in setences.split(' '):
            word_dict[word] += 1

    return word_dict


def add_features(pd):
    # TODO: Come up with more features.
    # import pdb; pdb.set_trace()
    pd['total_length'] = pd['question_text'].apply(len)
    pd['capitals'] = pd['question_text'].apply(
        lambda x: sum(1 for c in x if c.isupper()))
    pd['caps_ratio'] = pd.apply(lambda x: float(
        x['capitals'])/float(x['total_length']), axis=1)
    pd['num_words'] = pd['question_text'].str.count('\S+')
    pd['num_unique_words'] = pd['question_text'].apply(
        lambda x: len(set(x.split(' '))))
    pd['unique_word_ratio'] = 1.0 * pd['num_unique_words'] / pd['num_words']


def prepare_data():
    pd_data = get_data()
    add_features(pd_data)
    train_data, test_data_all, train_label, test_label_all = train_test_split(
        pd_data['question_text'], pd_data['target'], test_size=0.4, shuffle=False)

    all_features = pd_data[['total_length', 'num_words',
                            'caps_ratio', 'unique_word_ratio']].fillna(0)
    train_features, test_features_all = train_test_split(
        all_features, test_size=0.4, shuffle=False)

    validation_data, test_data, validation_label, test_label = train_test_split(
        test_data_all, test_label_all, test_size=0.5, shuffle=False)
    validation_features, test_features = train_test_split(
        test_features_all, test_size=0.5, shuffle=False)

    ss = StandardScaler()
    ss.fit(np.vstack((train_features, validation_features, test_features)))
    train_features = ss.transform(train_features)
    validation_features = ss.transform(validation_features)
    test_features = ss.transform(test_features)

    word_dict = build_word_dict(pd_data)

    return train_data, validation_data, test_data, train_label, validation_label, test_label, train_features, validation_features, test_features, len(word_dict)


@timer
def fit_and_predict(load_test_data,
                    train_data,
                    validation_data,
                    test_data,
                    train_label,
                    validation_label,
                    test_label,
                    train_features,
                    validation_features,
                    test_features,
                    word_count,
                    if_plt_roc):

    tokenizer = Tokenizer(num_words=120000)
    tokenizer.fit_on_texts(list(train_data))
    train_data = tokenizer.texts_to_sequences(train_data)
    validation_data = tokenizer.texts_to_sequences(validation_data)
    test_data = tokenizer.texts_to_sequences(test_data)

    # Pad the train and test data. This is important otherwise the dimention of the inputs won't match!
    train_data = pad_sequences(train_data, maxlen=64)
    validation_data = pad_sequences(validation_data, maxlen=64)
    test_data = pad_sequences(test_data, maxlen=64)
    # import pdb; pdb.set_trace()
    embedding_data = EmbeddingLayer(tokenizer.word_index, word_count)
    cnn_model = CNNModel(embedding_size=300, hidden_size_1=96, num_layers_1=1, hidden_size_2=96,
                         num_layers_2=1, dense_size_1=388, dense_size_2=24, output_size=1, embeddings=embedding_data)

    if CUDA:
        cnn_model.cuda()

    num_params = 0
    for params in cnn_model.parameters():
        if params.requires_grad:
            num_params += np.prod(params.size())
    print(f'Num of params to train: {num_params}')

    # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
    batch_size = 100
    n_iters = 3000
    num_epochs = 20
    print(f'Num of epochs: {num_epochs}')

    train_data_cuda = torch.tensor(train_data, dtype=torch.long).cuda(
    ) if CUDA else torch.tensor(train_data, dtype=torch.long)
    train_features_cuda = torch.tensor(train_features, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(train_features, dtype=torch.float)

    validation_data_cuda = torch.tensor(validation_data, dtype=torch.long).cuda(
    ) if CUDA else torch.tensor(validation_data, dtype=torch.long)
    validation_features_cuda = torch.tensor(validation_features, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(validation_features, dtype=torch.float)

    test_data_cuda = torch.tensor(test_data, dtype=torch.long).cuda(
    ) if CUDA else torch.tensor(test_data, dtype=torch.long)
    test_features_cuda = torch.tensor(test_features, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(test_features, dtype=torch.float)

    train_label_cuda = torch.tensor(train_label.values, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(train_label.values, dtype=torch.float)
    train = torch.utils.data.TensorDataset(
        train_data_cuda, train_features_cuda, train_label_cuda)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False)

    validation_label_cuda = torch.tensor(validation_label.values, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(validation_label.values, dtype=torch.float)
    validation = torch.utils.data.TensorDataset(
        validation_data_cuda, validation_features_cuda, validation_label_cuda)
    validation_loader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, shuffle=False)

    test_label_cuda = torch.tensor(test_label.values, dtype=torch.float).cuda(
    ) if CUDA else torch.tensor(test_label.values, dtype=torch.float)
    test = torch.utils.data.TensorDataset(
        test_data_cuda, test_features_cuda, test_label_cuda)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False)

    # test_label_cuda = torch.tensor(test_label, dtype=torch.float).cuda() if CUDA else torch.tensor(test_label, dtype=torch.float)

    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    iter = 0
    previous_loss = None

    for epoch in range(num_epochs):
        print(f'Start epoch: {epoch}')
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        optimizer.zero_grad()
        cnn_model.train()

        train_loss = 0
        for my_train_data, my_train_features, my_train_labels in tqdm(train_loader, disable=True):
            outputs = cnn_model(my_train_data, my_train_features)
            # import pdb; pdb.set_trace()
            loss = criterion(outputs[:, 0], my_train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        print(f'My train loss is: {train_loss}')

        cnn_model.eval()
        validation_loss = 0
        validation_outputs = np.zeros(len(validation_data))

        for i, (my_validation_data, my_validation_features, my_validation_labels) in enumerate(validation_loader):
            validation_outputs = cnn_model(
                my_validation_data, my_validation_features)

            my_loss = criterion(validation_outputs[:, 0], my_validation_labels)
            validation_loss += my_loss.item() / len(validation_loader)

        print(f'My validation loss is {validation_loss}')

        if epoch >= 5:
            # Calculate Accuracy
            # import pdb; pdb.set_trace()
            # Iterate through test dataset
            all_predicted_label = np.zeros(len(test_label.values))
            # To avoid CUDA OOM, have to batch even for testset data.
            for i, (my_test_data, my_test_features, _) in enumerate(test_loader):
                outputs = cnn_model(my_test_data, my_test_features)
                predicted = (outputs[:, 0] > 0.5).float()

                # Total number of labels
                # total = len(test_loader)
                # import pdb; pdb.set_trace()
                if CUDA:
                    predicted_label = predicted.detach().cpu().numpy()
                else:
                    predicted_label = predicted.numpy()

                all_predicted_label[i * batch_size: (i+1) * batch_size] = predicted_label

            # import pdb; pdb.set_trace()
            print(f'accuracy is: {accuracy_score(test_label.values, all_predicted_label)}')
            print(f'f1 score is: {f1_score(test_label.values, all_predicted_label)}')
            print(f'confusion_matrix score is: {confusion_matrix(test_label.values, all_predicted_label)}')

    # if load_test_data:
    #   del test_label_OR_test_data['question_text']

    #   test_label_OR_test_data.insert(1, 'prediction', prediction)
    #   test_label_OR_test_data.to_csv('submission.csv', index=False)
    #   return prediction
    # else:
    #   if if_plt_roc:
    #       plt_roc(test_label_OR_test_data, prediction)

    #   print(f'accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
    #   print(f'f1 score is: {f1_score(test_label_OR_test_data, prediction)}')
    #   print(f'confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction)}')


if __name__ == "__main__":
    load_test_data = False

    train_data, validation_data, test_data, train_label, validation_label, test_label, train_features, validation_features, test_features, word_count = prepare_data()

    fit_and_predict(load_test_data,
                    train_data,
                    validation_data,
                    test_data,
                    train_label,
                    validation_label,
                    test_label,
                    train_features,
                    validation_features,
                    test_features,
                    word_count,
                    True)
