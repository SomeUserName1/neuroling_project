import argparse
import time

from sklearn.model_selection import train_test_split
from config import *
from models.Handcrafted import MediumConv1DNet, DeepDenseNet, WideDenseNet, SmallDenseNet
from models.NAS import NAS
from util import preprocess
from util.visualize import visualize
import keras.backend as K


def main(data_path, weight_path, verbose, frequency, electrodes, search_time):
    """
    Args:
        data_path: path where data is located
        weight_path: path to log and save weights to
        verbose: wether training should output progress and metrics
        electrodes: which electrodes to use
        search_time: how much time to take for searching an architecture
        frequency: which frequency to use
    """
    start = time.time()
    spectra, scores = preprocess.preprocess(data_path, electrodes, frequency)
    end = time.time()
    print("Importing data from mat files finished! Took %.3f s" % (end - start))

    print(preprocess.mean_cov(spectra, scores))

    for net in [SmallDenseNet, MediumConv1DNet, WideDenseNet, DeepDenseNet]:  # , NAS]:
        K.clear_session()
        train_x, test_x, train_y, test_y = train_test_split(spectra, scores, shuffle=False, train_size=0.90)

        if net is not NAS:
            instance = net(os.path.sep.join([weight_path, str(frequency)]), frequency, electrodes)
            instance.build()
            instance.fit(train_x, train_y)
            instance.evaluate(test_x, test_y)
        else:
            instance = NAS(verbose=verbose, path=os.path.sep.join([weight_path, str(frequency), "NAS"]),
                           training_time=search_time)
            instance.fit(spectra, scores, time_limit=search_time)
            instance.final_fit(train_x, train_y, test_x, test_y, trainer_args={'max_no_improvement_num': 30},
                               retrain=False)
            visualize(os.path.sep.join([weight_path, str(frequency), "NAS"]))
            print(instance.evaluate(test_x, test_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=DATA_SET_DIR, type=str)
    parser.add_argument("--output_path", default=MODEL_OUT_DIR, type=str)
    parser.add_argument("--verbose", default=VERBOSE, type=bool)
    parser.add_argument("--frequency", default=FREQUENCY, type=int)
    parser.add_argument("--electrodes", default=ELECTRODES, type=list)
    parser.add_argument("--search_time", default=SEARCH_TIME, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception("Data set path given does not exists")

    main(args.data_path, args.output_path, args.verbose, args.frequency, args.electrodes, args.search_time)
