import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, plot_training_curve
import argparse
import pickle

# argparser
parser = argparse.ArgumentParser(description='Hyperparameters for the experiment')
parser.add_argument('--model_type', action="store", dest="model_type", type=str, default='hybrid_pqc')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.1)
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=20)
args = parser.parse_args()


def prepare_classic_dataset():
    # load dataset
    x = np.load('data_mfpt_3_classes.npz')['x']
    y = np.load('data_mfpt_3_classes.npz')['y']
    # separate into test and train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    # normalize
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # one-hot encode
    y_train = np.eye(3)[y_train]
    y_test = np.eye(3)[y_test]

    return x_train, x_test, y_train, y_test


def prepare_quantum_dataset(x):
    # Angle encoding
    # Qubits
    q = cirq.GridQubit.rect(1, len(x))
    # Operations
    ops = [cirq.ry(2 * x[i]).on(q[i]) for i in range(len(x))]
    # 1 circuit == 1 datapoint
    circuit = cirq.Circuit(ops)

    return circuit


def simple_hybrid_model():
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


def nn_model():
    # The classical neural network layers.
    nn = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='elu'),
                              tf.keras.layers.Dense(3, activation='softmax')])

    return nn


def train(model, learning_rate, epochs, data='quantum'):
    # load & prepare dataset
    x_train, x_test, y_train, y_test = prepare_classic_dataset()
    if data == 'quantum':
        x_train = tfq.convert_to_tensor([prepare_quantum_dataset(x) for x in x_train])
        x_test = tfq.convert_to_tensor([prepare_quantum_dataset(x) for x in x_test])
    else:
        pass

    # Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=epochs,
                        verbose=1)

    # predict
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Evaluate
    results_train = model.evaluate(x_train, y_train)
    results_test = model.evaluate(x_test, y_test)

    # confusion matrix
    y_test_ne = np.argmax(y_test, axis=1)
    y_test_pred_ne = np.argmax(y_test_pred, axis=1)
    cm = confusion_matrix(y_test_ne, y_test_pred_ne)
    return {'train_accuracy': results_train[1],
            'test_accuracy': results_test[1],
            'train_loss': results_train[0],
            'test_loss': results_test[0],
            'cm': cm,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'history': history.history}


if __name__ == "__main__":

    # define parameters
    if args.model_type == 'simple_hybrid_model':
        model = simple_hybrid_model()
        data = 'quantum'
    elif args.model_type == 'nn_model':
        model = nn_model()
        data = 'classic'
    else:
        raise Exception('Model type not recognized.')

    # run model
    output = train(model, args.learning_rate, args.epochs, data)

    # save plots
    plot_confusion_matrix(output['cm'],
                          ['ND', 'OR', 'IR'],
                          title='MFPT Classification',
                          normalize=True)
    plot_training_curve(output['history']['loss'], 'Loss')
    plot_training_curve([100 * i for i in output['history']['accuracy']], 'Accuracy [%]')

    # print some metrics of the run
    print(f'{"train_accuracy"}: {round(100 * output["train_accuracy"], 2)}')
    print(f'{"test_accuracy"}: {round(100 * output["test_accuracy"], 2)}')

    # save output as pkl}
    print(output)
    output_file = open("output.pkl", "wb")
    pickle.dump(output, output_file)
    output_file.close()
