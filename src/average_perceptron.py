from shogun import RealFeatures, BinaryLabels, AveragedPerceptron, AccuracyMeasure, F1Measure

from data_pre_processing import load_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_PATH = 'preprocess_tokenize_and_stem_new.csv'
TEST_PATH = 'test.csv'

def prepare_data(load_test_data=False):
    pd_data = load_data(TRAIN_PATH)
    vector = TfidfVectorizer("english")

    feature_matrics = vector.fit_transform(pd_data['question_text'].values.astype('U'))

    # shuffle=False means pick the last 20% as dev data set.
    if load_test_data:
        test_data = load_data(TEST_PATH)
        test_feature_matrics = vector.transform(test_data['question_text'].values.astype('U'))
        return feature_matrics, test_feature_matrics, pd_data['target'], test_data
    else:
        return train_test_split(feature_matrics, pd_data['target'], test_size=0.2, shuffle=False)

def fit_and_predict(load_test_data, 
                    train_data, 
                    test_feature_matrics, 
                    train_label, 
                    test_label_OR_test_data):
    features_train = RealFeatures(train_data)
    features_test = RealFeatures(test_feature_matrics)
    labels_train = BinaryLabels(train_label)

    learn_rate = 1.0
    max_iter = 1000
    perceptron = AveragedPerceptron(features_train, labels_train)
    perceptron.set_learn_rate(learn_rate)
    perceptron.set_max_iter(max_iter)
    perceptron.train()
    perceptron.set_features(features_test)
    labels_predict = perceptron.apply()    
    if load_test_data:
        del test_label_OR_test_data['question_text']
        # import pdb; pdb.set_trace()
        test_label_OR_test_data.insert(1, 'prediction', prediction80)
        test_label_OR_test_data.to_csv('submission.csv', index=False)
        return prediction
    else:
        labels_test = BinaryLabels(test_label_OR_test_data)
        accEval = AccuracyMeasure()
        accuracy = accEval.evaluate(labels_predict, labels_test)
        f1Eval = F1Measure()
        f1_score = f1Eval.evaluate(labels_predict, labels_test)
        print('#accuracy is: ', accuracy)
        print('#F1 score is: ', f1_score)

if __name__ == "__main__":
    load_test_data = False
    train_data, test_data, train_label, test_label = prepare_data(load_test_data)
    fit_and_predict(load_test_data, train_data, test_data, train_label, test_label)

