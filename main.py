import scipy.io as sio
import argparse
import os

def parse_mat_files(path, export_to_csv):
    coherence_spectra = []
    comprehension_scores = []
    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(mat_file)
        for i in range(1,3):
            session_i = mat_contents['conn'][0, 0][i]
            coherence_spectra.concatenate = session_i['cohspectrm']
            comprehension_scores.concatenate = session_i['trialinfo'][5]
        return coherence_spectra, comprehension_scores


def main(shape_predictor_path):
    parse_mat_files('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO final interface
    parser.add_argument("--shape_predictor_path", default=0, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_set_dir):
        raise Exception("Data set path given does not exists")

    main(args.shape_predictor_path)