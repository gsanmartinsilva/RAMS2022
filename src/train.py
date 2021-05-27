import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, plot_training_curve
import argparse
import os
import random


# argparser
parser = argparse.ArgumentParser(description='Hyperparameters for the experiment')
parser.add_argument('--model_type', action="store", dest="model_type", type=str, default='hybrid_pqc')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.1)
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=20)
parser.add_argument('--n_runs', action="store", dest="n_runs", type=int, default=10)
args = parser.parse_args()


# set seeds
def set_seeds(s):
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


class QML:
    def __init__(self, model_id, run_number):
        self.model_id = model_id
        self.run_number = run_number
        if self.model_id == 'simple_hybrid_pqc':
            self.model = self._simple_hybrid_model()
        else:
            raise Exception('Model not specified.')

    def _angle_encoding(self, x):
        # Qubits
        q = cirq.GridQubit.rect(1, len(x))
        # Operations
        ops = [cirq.ry(2 * x[i]).on(q[i]) for i in range(len(x))]
        # 1 circuit == 1 datapoint
        circuit = cirq.Circuit(ops)

        return circuit

    def _simple_hybrid_model(self):
        # Parameters
        params = sympy.symbols('a b c')
        # Qubits
        q = cirq.GridQubit.rect(1, 5)
        # Operations
        ops = []
        for i in range(5):
            ops.append(cirq.ry(params[0]).on(q[i]))
            ops.append(cirq.rx(params[1]).on(q[i]))
            ops.append(cirq.rz(params[2]).on(q[i]))
        # PQC circuit model
        model_circuit = cirq.Circuit(ops)
        # The classical neural network layers.
        nn = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='elu'),
                                  tf.keras.layers.Dense(3, activation='softmax')])
        # Circuit's input
        circuit_input = tf.keras.Input(shape=(), dtype=tf.string, name='circuits_input')
        # TFQ layer for circuits.
        measurement_ops = [cirq.Z(q[i]) for i in range(5)]
        circuit_layer = tfq.layers.PQC(model_circuit, measurement_ops)
        # The Keras model
        model = tf.keras.Model(inputs=circuit_input, outputs=nn(circuit_layer(circuit_input)))

        return model

    def _simple_nn_model(self):
        # The classical neural network layers.
        model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='elu'),
                                     tf.keras.layers.Dense(3, activation='softmax')])

        return model

    def prepare_classic_dataset(self, data):
        # load dataset
        x = data['x']
        y = data['y']
        # separate into test and train
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        # normalize
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        # one-hot encode
        y_train = np.eye(3)[y_train]
        y_test = np.eye(3)[y_test]

        return x_train, x_test, y_train, y_test

    def prepare_quantum_dataset(self, x_train, x_test):
        x_train_q = tfq.convert_to_tensor([self._angle_encoding(xi) for xi in x_train])
        x_test_q = tfq.convert_to_tensor([self._angle_encoding(xi) for xi in x_test])

        return x_train_q, x_test_q

    def train(self, x_train, y_train, learning_rate, epochs):

        # Training
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 epochs=epochs,
                                 verbose=1)

        return history

    def evaluate(self, x_train, x_test, y_train, y_test):
        # predict
        _ = self.model.predict(x_train)
        y_test_pred = self.model.predict(x_test)

        # Evaluate
        results_train = self.model.evaluate(x_train, y_train)
        results_test = self.model.evaluate(x_test, y_test)

        # confusion matrix
        y_test_ne = np.argmax(y_test, axis=1)
        y_test_pred_ne = np.argmax(y_test_pred, axis=1)
        cm = confusion_matrix(y_test_ne, y_test_pred_ne)

        return {'train_accuracy': results_train[1],
                'test_accuracy': results_test[1],
                'train_loss': results_train[0],
                'test_loss': results_test[0],
                'cm': cm,
                'run': self.run_number}


if __name__ == "__main__":
    results = []
    # read seeds
    seeds = open("seeds.txt", "r").readlines()
    for n in range(args.n_runs):
        print(f'\nRun {n + 1} with seed: {seeds[n]}')
        set_seeds(int(seeds[n]))
        # generate an instance of the model
        m = QML(args.model_type, n)
        # load dataset
        data = np.load('../data/data_mfpt_3_classes.npz')
        # transform the dataset
        x_train, x_test, y_train, y_test = m.prepare_classic_dataset(data)
        x_train, x_test = m.prepare_quantum_dataset(x_train, x_test)
        # train the model
        h = m.train(x_train, y_train, args.learning_rate, args.epochs)
        # save figures for evolution curves
        plot_training_curve(h.history['loss'],
                            'Loss', n,
                            '../docs/figures/Loss_run')
        plot_training_curve([100*i for i in h.history['accuracy']],
                            'Accuracy [%]', n,
                            '../docs/figures/Acc_run')
        # evaluate model and generate dictionary
        d = m.evaluate(x_train, x_test, y_train, y_test)
        # save confusion matrix
        plot_confusion_matrix(d['cm'],
                              ['ND', 'OR', 'IR'], n,
                              title='../docs/figures/MFPT Classification Case Study: Test Data',
                              normalize=True)
        # generate pd.DataFrame with results
        d.pop('cm')
        results.append(d)

    # print dictionary to console and save it to logs as csv
    df = pd.DataFrame(data=results)
    print('\n\n\nRaw results:\n', df.to_string())
    print('\n\n\nAverage results:\n', df.describe().to_string())
    # save results
    df.to_csv(f'../docs/results/{args.model_type}_results.csv', sep=';')
    df.describe().to_csv(f'../docs/results/{args.model_type}_average_results.csv', sep=';')
