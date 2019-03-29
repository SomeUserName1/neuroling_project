import argparse
import os
import time

from sklearn.model_selection import KFold, train_test_split

from models.NAS import NAS
from models.Handcrafted import MediumConv1DNet, DeepDenseNet, WideDenseNet, SmallDenseNet
from util import preprocess
from util.visualize import visualize


def main(data_path, weight_path, verbose, electrodes, search_time):
    """
    Args:
        data_path:
        weight_path:
        verbose:
        electrodes:
        search_time:
    """
    for i in ['5', '10', 'all']:
        if i == '5':
            frequency = 5
        elif i == '10':
            frequency = 10
        else:
            frequency = None

        start = time.time()
        spectra, scores = preprocess.preprocess(data_path, electrodes, frequency, False)
        end = time.time()
        print("Importing data from mat files finished! Took %.3f s" % (end - start))

        for net in [MediumConv1DNet, SmallDenseNet, WideDenseNet, DeepDenseNet]: #, NAS]:
            print(i, net)
            train_x, test_x, train_y, test_y = train_test_split(spectra, scores, shuffle=False, train_size=0.95)

            if net is not NAS:
                instance = net(os.path.sep.join([weight_path, i]), frequency, electrodes)
                instance.build()
                instance.fit(train_x, train_y)
                instance.evaluate(test_x, test_y)
            else:
                instance = NAS(verbose=verbose, path=os.path.sep.join([weight_path, i, "NAS"]),
                               training_time=search_time)
                instance.fit(spectra, scores, time_limit=search_time)
                instance.final_fit(train_x, train_y, test_x, test_y, trainer_args={'max_no_improvement_num': 30},
                                   retrain=False)
                visualize(os.path.sep.join([weight_path, i, "NAS"]))
                print(instance.evaluate(test_x, test_y))


if __name__ == "__main__":
    ALL_ELECTRODES = ['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4',
                      'CP5', 'CP6',
                      'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
                      'FC6', 'FCZ', 'FP1',
                      'FP2', 'FT10', 'FT7', 'FT8', 'FT9', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
                      'P7', 'P8',
                      'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']
    CLUSTER = ['AFZ', 'C2', 'C4', 'CP4', 'CP6', 'F1']

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        default='C:\\Users\\Fabi\\ownCloud\\workspace\\uni\\7\\neuroling\\neuroling_project\\data\\v1',
                        # default='data/v1',
                        type=str)
    parser.add_argument("--weight_path",
                        default=os.sep.join(['C:', 'Users', 'Fabi', 'ownCloud', 'workspace', 'uni', '7', 'neuroling',
                                             'neuroling_project', 'models', 'weights']),
                        # default='models/weights',
                        type=str)
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--frequency", default=10, type=int)
    parser.add_argument("--electrodes", default=ALL_ELECTRODES, type=list)
    parser.add_argument("--search_time", default=30, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception("Data set path given does not exists")

    main(args.data_path, args.weight_path, args.verbose, args.electrodes,
         args.search_time)
