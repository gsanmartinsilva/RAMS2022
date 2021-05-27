import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_training_curve(y, y_label, n, title='plot'):
    plt.figure(figsize=(5, 5))
    plt.plot(y, marker='v', markersize=2)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + f'_{n}' + '.svg')


def plot_confusion_matrix(cm, classes, n, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, show=False):

    """
    DESCRIPCION: Funcion que sirve para plotear una matriz de confusion.
    INPUT:
    (cm)         Matriz de confusion (n-darray)
    (classes)    Nombre de las clases, lista de strings con el nombre de cada clase.
    (title)      Titulo de la matriz de confusion, string.
    (cmap)       Color de la matriz de confusion
    """
    plt.figure(figsize=(10, 10))
    # plt.title(title, fontsize=35)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # We change the fontsize of minor ticks label
    plt.tick_params(axis='both', which='major', labelsize=25)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, decimals=3)
    else:
        cm = np.trunc(cm).astype(int)

    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        if normalize:
            plt.text(j, i, '{:.1%}'.format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=35,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     fontsize=35,
                     color="white" if cm[i, j] > thresh else "black")

    # plt.colorbar()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)
    plt.ylabel('True label', fontsize=35)
    plt.xlabel('Predicted label', fontsize=35)
    plt.tight_layout()
    plt.savefig(title + f'_{n}' + '.svg')
