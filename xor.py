import numpy
import keras

def main():
    train_data = numpy.array([[0,0], [0,1], [1,0], [1,1]], "float32")
    valid_data = numpy.array([[  0], [  1], [  1], [  0]], "float32")

    model = keras.models.Sequential()
    model.add(keras.layers.core.Dense(32, input_dim = 2, activation = 'relu'))
    model.add(keras.layers.core.Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['binary_accuracy'])

    model.fit(train_data, valid_data, epochs = 16000, verbose = 0, batch_size = 16)

    print(model.predict(train_data))

if __name__ == "__main__":
    main()
