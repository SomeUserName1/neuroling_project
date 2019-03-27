from bisect import bisect_right, bisect_left

import autokeras
import scipy.io as sio
from autokeras.search import BayesianSearcher
from keras import Input, Model, losses, optimizers
from keras.layers import Conv1D, Dense, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from nas.random import RandomSearcher
from sklearn.model_selection import KFold, cross_val_score
from models import EEGLangComprehensionNAS
import argparse
import os
import shutil
import numpy as np
import time
from autokeras.utils import pickle_from_file
from graphviz import Digraph
from nas.random import RandomSearcher
from autokeras.search import BayesianSearcher

from models.vgg import EEGLangComprehensionVGG


def parse_mat_files(path, electrode_list, export_to_csv):
    coherence_spectra = []
    comprehension_scores = []
    frequencies = []

    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            electrodes = []
            cluster_indexes = []
            session_i = mat_contents['conn'][0, i][0]

            label_struct = session_i['labelcmb'][0]
            for label in label_struct:
                electrodes = np.append(electrodes, label[0])

            for el in electrode_list:
                cluster_indexes.append(np.where(electrodes == el)[0][0])

            coherence_spectra.append(session_i['cohspctrm'][0][:, cluster_indexes, :])

            for j in range(0, len(session_i['trialinfo'][0])):
                comprehension_scores.append(session_i['trialinfo'][0][j][2])

            if len(frequencies) == 0:
                frequencies = session_i['freq']
                frequencies = np.concatenate([x for x in frequencies]).flatten()

    coherence_spectra = np.concatenate([x for x in coherence_spectra])
    comprehension_scores = np.array(comprehension_scores)

    if export_to_csv:
        coherence_spectra.tofile(path + '\\..\\coherence_spectra.csv', sep=',')
        comprehension_scores.tofile(path + '\\..\\comprehension_scores.csv', sep=',')
        frequencies.tofile(path + '\\..\\frequencies.csv', sep=',')

    return coherence_spectra, comprehension_scores, frequencies


def get_electrodes(path):
    all_electrodes = []
    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            electrodes = []
            session_i = mat_contents['conn'][0, i][0]

            label_struct = session_i['labelcmb'][0]
            for label in label_struct:
                electrodes = np.append(electrodes, label[0])
            all_electrodes.append(electrodes)
    return all_electrodes


def eval_classifier(classifier_i, spectra, scores, search_time):

    classifier_i.fit(spectra, scores, time_limit=search_time)

    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(spectra):
        train_x, test_x = spectra[train_index], spectra[test_index]
        train_y, test_y = scores[train_index], scores[test_index]

        classifier_i.final_fit(train_x, train_y, test_x, test_y, trainer_args={'max_no_improvement_num': 30}, retrain=True)
        print(classifier_i.evaluate(test_x, test_y))


def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
        to_pdf(graph, os.path.join(path, str(model_id)))

def preprocess(x, frequencies, frequency_component):
    if frequency_component is not None:
        if frequency_component < 1 or frequency_component > 50:
            raise Exception("Select a frequency between 1 and 50 Hz!")
        else:
            return np.squeeze(x[:, :, frequency_lookup(frequencies, frequency_component)])
    else:
        return x

def frequency_lookup(frequencies, frequency_component):
    lower = frequency_component - 1
    upper = frequency_component + 1

    idx_lower = bisect_right(frequencies, lower)
    idx_upper = bisect_left(frequencies, upper)

    return [range(idx_lower, idx_upper + 1)]

def build():
    """

    Returns:
        An instance of the EmoPy VGG Face model
    """
    # x = VGGFace(include_top=False, input_shape=self.input_shape)
    a = Input((64,5))
    x = Conv1D(16,5)(a)
    x = Conv1D(8, 1)(x)
    x = Flatten()(x)
    x = Dense(4, activation='sigmoid', kernel_initializer='normal',name='fc0')(x)
    x = Dense(2, activation='sigmoid',kernel_initializer='normal', name='fc1')(x)
    b = Dense(1, activation='sigmoid',kernel_initializer='normal', name='fc2')(x)
    print("VGG")
    model = Model(a, b)
    model.summary()

    model.compile(loss=losses.mean_squared_error,
                       optimizer=optimizers.Adam(0.1),
                       metrics=['mse'])
    return model


def NAS(spectra, scores, frequencies, weight_path, verbose, frequency, searcher, search_time):
    if os.path.exists(weight_path):
        shutil.rmtree(weight_path)

    classifier_i = EEGLangComprehensionNAS.EEGLangComprehensionNAS(weight_path, frequencies, frequency, verbose=verbose,
                                                                   search_type=searcher)
    eval_classifier(classifier_i, spectra, scores, search_time)
    visualize(weight_path)


def plain(spectra, scores, frequencies, frequency):
    spectra = preprocess(spectra, frequencies, frequency)
    estimator = KerasRegressor(build_fn=build, epochs=100, batch_size=5, verbose=1)
    kfold = KFold(n_splits=10, random_state=42)
    results = cross_val_score(estimator, spectra, scores, cv=kfold, verbose=1, n_jobs=-1)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def main(data_path, weight_path, verbose, frequency, electrodes, searcher, search_time):
    start = time.time()
    spectra, scores, frequencies = parse_mat_files(data_path, electrodes, False)
    end = time.time()
    print("Importing data from mat files finished! Took %.3f s" % (end - start))
    print("Data shapes: spectre", spectra.shape, "scores: ", scores.shape)

    #NAS(spectra, scores, frequencies, weight_path, verbose, frequency, searcher, search_time)
    plain(spectra, scores, frequencies, frequency)


if __name__ == "__main__":
    ALL_ELECTRODES = ['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                      'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCZ', 'FP1',
                      'FP2', 'FT10', 'FT7', 'FT8', 'FT9', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                      'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']
    CLUSTER = ['AFZ', 'C2', 'C4', 'CP4', 'CP6', 'F1']

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        default='C:\\Users\\Fabi\\ownCloud\\workspace\\uni\\7\\neuroling\\neuroling_project\\data\\v1',
                        # default='data/v1',
                        type=str)
    parser.add_argument("--weight_path",
                default='C:\\Users\\Fabi\\ownCloud\\workspace\\uni\\7\\neuroling\\neuroling_project\\models\\weights10',
                        # default='models/weights',
                        type=str)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--frequency", default=10, type=int)
    parser.add_argument("--electrodes", default=ALL_ELECTRODES, type=list)
    parser.add_argument("--search_time", default=10 * 60, type=int)
    parser.add_argument("--searcher", default=BayesianSearcher, type=autokeras.search.Searcher) # RandomSearcher

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception("Data set path given does not exists")

    main(args.data_path, args.weight_path, args.verbose, args.frequency, args.electrodes, args.searcher,
         args.search_time)
