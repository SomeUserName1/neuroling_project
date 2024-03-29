import os

from autokeras.utils import pickle_from_file
from graphviz import Digraph
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def to_pdf(graph, path):
    """
        prints the dot/graphviz graph to a pdf file
    Args:
        graph: dot graph
        path: path to write the pdf to
    """
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    """
        constructs a dot graph, visualizing the found model architectures
    Args:
        path: of the model files
    """
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
        to_pdf(graph, os.path.join(path, str(model_id)))


def plot_history(history,out_dir):
    """
        Plots the training loss and metrics of a trained classifier suring training
    Args:
        history: the history object returned from training the model
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.clf()
    plt.figure()
    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error ')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(out_dir)


def plot_predictions(test_labels, test_predictions, out_dir):
    """
    plots the predictions and true labels
    Args:
        test_labels: true labels
        test_predictions: values predicted by the regressor
    """
    plt.clf()
    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    x = np.linspace(0, 1, 100)
    plt.plot(x, x + 0, linestyle='-.')
    plt.ylim(0, 1)

    plt.savefig(out_dir)
