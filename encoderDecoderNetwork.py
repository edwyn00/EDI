from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DatasetManager import *



class Encoder(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=dims[0],
            #activation=tf.nn.prelu,
            #kernel_initializer='he_uniform'
        )
        self.encoder_prelu_1 = tf.keras.layers.PReLU()
        self.encoder_dropout_1 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=dims[1],
            activation=tf.nn.relu
        )
        self.output_layer = tf.keras.layers.Dense(
            units=dims[2],
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        activation = self.encoder_prelu_1(activation)
        activation = self.encoder_dropout_1(activation)
        #activation = self.hidden_layer2(activation)
        return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=dims[0],
            #activation=tf.nn.prelu,
            #kernel_initializer='he_uniform'
        )
        self.decoder_prelu_1 = tf.keras.layers.PReLU()
        self.decoder_dropout_1 = tf.keras.layers.Dropout(0.2)
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=dims[1],
            activation=tf.nn.relu
        )
        self.output_layer = tf.keras.layers.Dense(
            units=dims[2],
            #activation=tf.nn.relu
        )

    def call(self, code):
        activation = self.hidden_layer(code)
        activation = self.decoder_prelu_1(activation)
        activation = self.decoder_dropout_1(activation)
        #activation = self.hidden_layer2(activation)
        return self.output_layer(activation)

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder_dims, decoder_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoder_dims)
        self.decoder = Decoder(decoder_dims)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed



    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

def loss(model, x_train, y_train):
    intermidiate_loss = tf.square(tf.subtract(model(x_train), y_train))
    #print(intermidiate_loss)
    reconstruction_error = tf.reduce_mean(intermidiate_loss)
    return reconstruction_error


def train(loss, model, opt, x_train=None, y_train=None):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, x_train, y_train), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)

def main(learning_rate = 0.005, num_step_per_epoch = 500, epochs = 5, normal=True, seq_var=1, fields=None, test_size=5):

    dataset = DatasetManager(seq_var=seq_var, fields=fields)
    max_value = 100#seq_var+ dataset.tensorDim#dataset.max_value()
    dim_tensor = dataset.tensorDim
    print("Max Value for Normalization:", max_value)
    encoder_dims = [10, 8, 6]
    decoder_dims = [6, 10, dim_tensor]
    print("Encoder Dense Layer dimensions:", encoder_dims)
    print("Decoder Dense Layer dimensions:", decoder_dims)
    autoencoder = Autoencoder(encoder_dims, decoder_dims)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)


    writer = tf.summary.create_file_writer('tmp')
    tf.summary.trace_on(graph=True, profiler=True)

    with writer.as_default():
        #with tf.summary.record_if(True):
        tf.summary.trace_export(name="my_test", step=0, profiler_outdir='tmp')
        for epoch in range(epochs):
            dataset.new_epoch()
            print("\nEpoch number", epoch)
            for step in range(num_step_per_epoch):
                y_train, masks = dataset.next_batch()
                x_train = y_train #* masks
                if normal:
                    y_train /= max_value
                train(loss, autoencoder, opt, x_train=x_train, y_train=y_train)
                loss_values = loss(autoencoder, x_train, y_train)
                tf.summary.scalar('loss', loss_values, step=step + epoch*num_step_per_epoch)
                if step%10==0:
                    print("Loss for the step", step, "is", loss_values)
                    #print(y_train[0,:])
                    #print(x_train[0,:])

        test_with_masks = True
        test_size = test_size
        labels, masks = dataset.test_batch(test_size)
        test = np.copy(labels)
        if test_with_masks:
            test *= masks
        print(test.shape)
        print("______________________\nL'INPUT VECTOR\n______________________\n\n")
        print(test)
        print("______________________\nIL TARGET VECTOR\n______________________\n\n")
        print(labels)
        #test = np.expand_dims(test, axis=0)
        if normal:
            test /= max_value
        #test[0,2] = 0

        output = autoencoder(test)
        if normal:
            output *= max_value
            test*= max_value
        diff = output-labels
        diff_rounded = np.rint(diff)

        maskC = np.ones((test_size, dim_tensor))
        maskC -= masks
        diff_reconstructed = diff*maskC
        output = np.rint(output)
        print("______________________\nL'OUTPUT\n______________________\n\n")
        print(output)

        print("______________________\nLA DIFFERENZA ROUNDED\n______________________")
        print(diff_rounded)
        print("______________________\nLA DIFFERENZA RECONSSTRUCTED\n______________________")
        print(diff_reconstructed)
