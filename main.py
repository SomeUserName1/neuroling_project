import scipy.io as sio
import argparse
import os
import numpy as np
import time


def parse_mat_files(path, export_to_csv):
    coherence_spectra = []
    comprehension_scores = []
    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            session_i = mat_contents['conn'][0, i][0]
            coherence_spectra.append(session_i['cohspctrm'][0])
            for j in range(0,len(session_i['trialinfo'][0])):
                comprehension_scores.append(session_i['trialinfo'][0][j][2])

    coherence_spectra = np.concatenate([x for x in coherence_spectra])
    comprehension_scores = np.array(comprehension_scores)
    if export_to_csv:
        coherence_spectra.tofile(path + '\..\coherence_spectra.csv', sep=',')
        comprehension_scores.tofile(path + '\..\comprehension_scores.csv', sep=',')
    return coherence_spectra, comprehension_scores


def main(data_path):
    start = time.time()
    spectra, scores = parse_mat_files(data_path, True)
    end = time.time()
    print(end - start)
    print(spectra.shape)
    print(scores.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO final interface
    parser.add_argument("--data_path",
                        default='C:\\Users\\Fabi\\ownCloud\\workspace\\uni\\7\\neuroling\\neuroling_project\\data\\v1', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception("Data set path given does not exists")

    main(args.data_path)
