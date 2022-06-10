import keras.backend as K
from keras.layers import *
from keras.models import *
import numpy as np
import pickle
import os


class CNNModels:

    def __init__(self):
        self.models = []

    def __str__(self):
        return self.models

    def list_models(self):
        print(self.models)

    @staticmethod
    def one_secret_65_filters_encoder(input_size, activation, filter1, filter2, filter3):
        input_S1 = Input(shape=input_size)
        input_C = Input(shape=input_size)

        # Preparation Network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_3x3_1')(
            input_S1)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_4x4_1')(
            input_S1)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_5x5_1')(
            input_S1)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_5x5_1')(x)
        x1 = concatenate([x3, x4, x5])

        x = concatenate([input_C, x1])

        # Hiding network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hid0_3x3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hid0_4x4')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hid0_5x5')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hid1_3x3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hid1_4x4')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hid1_5x5')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hid2_3x3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hid2_4x4')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hid2_5x5')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hid3_3x3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hid3_4x4')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hid3_5x5')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hid4_3x3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hid4_4x4')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hid5_5x5')(x)
        x = concatenate([x3, x4, x5])

        output_Cprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_C')(x)

        return Model(inputs=[input_S1, input_C],
                     outputs=output_Cprime,
                     name='Encoder')

    @staticmethod
    def three_secret_65_filters_encoder(input_size, activation, filter1, filter2, filter3):
        input_S1 = Input(shape=input_size)
        input_S2 = Input(shape=input_size)
        input_S3 = Input(shape=input_size)
        input_C = Input(shape=input_size)

        # Preparation Network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_3x3_1')(
            input_S1)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_4x4_1')(
            input_S1)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_5x5_1')(
            input_S1)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_5x5_1')(x)
        x1 = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_3x3_2')(
            input_S2)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_4x4_2')(
            input_S2)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_5x5_2')(
            input_S2)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_5x5_2')(x)
        x2 = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_3x3_3')(
            input_S3)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_4x4_3')(
            input_S3)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep0_5x5_3')(
            input_S3)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_prep1_5x5_3')(x)
        x3__1 = concatenate([x3, x4, x5])

        x = concatenate([input_C, x1, x2, x3__1])

        # --------------------------------------------------------------------------------------------------------------
        # Hiding network
        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hide0_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hide0_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hide0_5x5_1')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hide1_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hide1_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hide1_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hide2_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hide2_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hide2_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hide3_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hide3_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hide3_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_hide4_3x3_3')(
            x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_hide4_4x4_3')(
            x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_hide4_5x5_3')(
            x)
        x = concatenate([x3, x4, x5])

        output_Cprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_C')(x)

        return Model(inputs=[input_S1, input_S2, input_S3, input_C],
                     outputs=output_Cprime,
                     name='Encoder')

    @staticmethod
    def one_secret_65_filters_decoder(input_size, activation, filter1, filter2, filter3):
        # Reveal network
        reveal_input = Input(shape=input_size)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_3x3_1')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_4x4_1')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_5x5_1')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev5_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        output_S1prime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_S1')(x)

        return Model(inputs=reveal_input, outputs=output_S1prime)

    @staticmethod
    def three_secret_65_filters_decoder_1(input_size, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=input_size)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_3x3_1')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_4x4_1')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_5x5_1')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_3x3_1')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_4x4_1')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev5_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        output_S1prime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_S1')(x)

        return Model(inputs=reveal_input, outputs=output_S1prime)

    @staticmethod
    def three_secret_65_filters_decoder_2(input_size, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=input_size)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise2')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_3x3_2')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_4x4_2')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_5x5_2')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_3x3_2')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_4x4_2')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev5_5x5_2')(x)
        x = concatenate([x3, x4, x5])

        output_S2prime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_S2')(x)

        return Model(inputs=reveal_input, outputs=output_S2prime)

    @staticmethod
    def three_secret_65_filters_decoder_3(input_size, activation, filter1, filter2, filter3):
        reveal_input = Input(shape=input_size)

        # Adding Gaussian noise with 0.01 standard deviation.
        input_with_noise = GaussianNoise(0.01, name='output_C_noise1')(reveal_input)

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_3x3_3')(
            input_with_noise)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_4x4_3')(
            input_with_noise)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev0_5x5_3')(
            input_with_noise)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev1_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev2_5x5_1')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev3_5x5_3')(x)
        x = concatenate([x3, x4, x5])

        x3 = Conv2D(filter1, (3, 3), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_3x3_3')(x)
        x4 = Conv2D(filter2, (4, 4), strides=(1, 1), padding='same', activation=activation, name='conv_rev4_4x4_3')(x)
        x5 = Conv2D(filter3, (5, 5), strides=(1, 1), padding='same', activation=activation, name='conv_rev5_5x5_3')(x)
        x = concatenate([x3, x4, x5])

        output_S3prime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation=activation, name='output_S3')(x)

        return Model(inputs=reveal_input, outputs=output_S3prime)

    def one_secret_65_filters_build_networks(self, input_size, rev_loss, full_loss, activation, filter1, filter2,
                                             filter3):
        input_S1 = Input(shape=input_size)
        input_C = Input(shape=input_size)

        encoder = self.one_secret_65_filters_encoder(input_size, activation, filter1, filter2, filter3)

        decoder = self.one_secret_65_filters_decoder(input_size, activation, filter1, filter2, filter3)
        decoder.compile(optimizer="adam", loss=rev_loss)
        decoder.trainable = False

        # Encoded Image
        output_Cprime = encoder([input_S1, input_C])
        # Decoded Image
        output_S1prime = decoder(output_Cprime)

        autoencoder = Model(inputs=[input_S1, input_C],
                            outputs=concatenate([output_S1prime, output_Cprime]))
        autoencoder.compile(optimizer='adam', loss=full_loss)

        return encoder, decoder, autoencoder

    def three_secret_65_filters_build_networks(self, input_size, rev_loss, full_loss, activation, filter1, filter2,
                                               filter3):
        secret_input_1 = Input(shape=input_size)
        secret_input_2 = Input(shape=input_size)
        secret_input_3 = Input(shape=input_size)
        cover_input = Input(shape=input_size)

        encoder = self.three_secret_65_filters_encoder(input_size, activation, filter1, filter2, filter3)

        decoder1 = self.three_secret_65_filters_decoder_1(input_size, activation, filter1, filter2, filter3)
        decoder1.compile(optimizer="adam", loss=rev_loss)
        decoder1.trainable = False

        decoder2 = self.three_secret_65_filters_decoder_2(input_size, activation, filter1, filter2, filter3)
        decoder2.compile(optimizer="adam", loss=rev_loss)
        decoder2.trainable = False

        decoder3 = self.three_secret_65_filters_decoder_3(input_size, activation, filter1, filter2, filter3)
        decoder3.compile(optimizer="adam", loss=rev_loss)
        decoder3.trainable = False

        output_Cprime = encoder([secret_input_1, secret_input_2, secret_input_3, cover_input])
        output_S1prime = decoder1(output_Cprime)
        output_S2prime = decoder2(output_Cprime)
        output_S3prime = decoder3(output_Cprime)

        autoencoder = Model(inputs=[secret_input_1, secret_input_2, secret_input_3, cover_input],
                            outputs=concatenate([output_S1prime, output_S2prime, output_S3prime, output_Cprime]))
        autoencoder.compile(optimizer='adam', loss=full_loss)

        return encoder, decoder1, decoder2, decoder3, autoencoder

    @staticmethod
    def lr_schedule(epoch_number):
        if 600 > epoch_number > 100:
            return 0.001
        elif epoch_number > 600:
            return 0.0001
        else:
            return 0.001

    def train_one_secret_65_filters(self, batch_size, epochs, path, shape, rev_loss, full_loss, secret_input,
                                    cover_input, verbose, save_interval, activation, filter1, filter2, filter3):

        """

        :param filter3:
        :param filter2:
        :param filter1:
        :param activation:
        :param batch_size:
        :param epochs: number of epochs
        :param path: weights model and loss history save path
        :param shape: input shape
        :param rev_loss: loss for network prep and hide
        :param full_loss: loss for autoencoder
        :param secret_input: secret image input
        :param cover_input: cover image input
        :param verbose: log in console or not (0, 1)
        :param save_interval: save every n epochs
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)

        encoder_model, decoder_model, autoencoder_model = self.one_secret_65_filters_build_networks(

            shape, rev_loss, full_loss, activation, filter1, filter2, filter3

        )

        history = []
        decode_loss = []
        auto_encode_loss = []
        size = secret_input.shape[0]

        for epoch in range(epochs):
            np.random.shuffle(secret_input)
            np.random.shuffle(cover_input)

            for i in range(0, size, batch_size):
                secret_batch = secret_input[i: min(i + batch_size, size)]
                cover_batch = cover_input[i: min(i + batch_size, size)]

                # hide network loss error term
                cover_loss = encoder_model.predict([secret_batch, cover_batch], verbose=verbose)

                auto_encode_loss.append(autoencoder_model.train_on_batch(x=[secret_batch, cover_batch],
                                                                         y=np.concatenate((secret_batch, cover_batch),
                                                                                          axis=3)))
                # reveal loss
                decode_loss.append(decoder_model.train_on_batch(x=cover_loss, y=secret_batch))
                K.set_value(autoencoder_model.optimizer.lr, self.lr_schedule(epoch))
                K.set_value(decoder_model.optimizer.lr, self.lr_schedule(epoch))

            if epoch + 1 == round(epochs / 4):
                encoder_model.save_weights(path + "encoder " + str(epoch))
                decoder_model.save_weights(path + "decoder " + str(epoch))
                autoencoder_model.save_weights(path + "autoencoder" + str(epoch))

            history.append(np.mean(auto_encode_loss))

        with open(path + "autoencoder_loss.pckl", 'wb') as f:
            pickle.dump(auto_encode_loss, f)

        with open(path + "reveal_loss.pckl", 'wb') as f:
            pickle.dump(decode_loss, f)

        with open(path + "loss_history.pckl", 'wb') as f:
            pickle.dump(history, f)

        return encoder_model, decoder_model, autoencoder_model

    def train_three_secret_65_filters(self,
                                      batch_size, epochs, path, shape,
                                      rev_loss, full_loss, secret1_input,
                                      secret2_input, secret3_input,
                                      cover_input, verbose, save_interval,
                                      activation, filter1, filter2, filter3
                                      ):

        """

        :param secret1_input:
        :param secret2_input:
        :param secret3_input:
        :param filter3:
        :param filter2:
        :param filter1:
        :param activation:
        :param batch_size:
        :param epochs: number of epochs
        :param path: weights model and loss history save path
        :param shape: input shape
        :param rev_loss: loss for network prep and hide
        :param full_loss: loss for autoencoder
        :param cover_input: cover image input
        :param verbose: log in console or not (0, 1)
        :param save_interval: save every n epochs
        :return:
        """

        if not os.path.exists(path):
            os.makedirs(path)

        encoder_model, decoder1_model, decoder2_model, decoder3_model, autoencoder_model = \
            self.three_secret_65_filters_build_networks(shape, rev_loss, full_loss, activation,
                                                        filter1, filter2, filter3)

        history = []
        decode1_loss = []
        decode2_loss = []
        decode3_loss = []
        auto_encode_loss = []
        size = secret1_input.shape[0]

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            np.random.shuffle(secret1_input)
            np.random.shuffle(secret2_input)
            np.random.shuffle(secret3_input)
            np.random.shuffle(cover_input)

            for i in range(0, size, batch_size):
                secret1_batch = secret1_input[i: min(i + batch_size, size)]
                secret2_batch = secret2_input[i: min(i + batch_size, size)]
                secret3_batch = secret3_input[i: min(i + batch_size, size)]
                cover_batch = cover_input[i: min(i + batch_size, size)]

                # hide network loss error term
                cover_loss = encoder_model.predict([secret1_batch, secret2_batch, secret3_batch, cover_batch],
                                                   verbose=verbose)

                auto_encode_loss.append(
                    autoencoder_model.train_on_batch(x=[secret1_batch, secret2_batch, secret3_batch, cover_batch],
                                                     y=np.concatenate(
                                                         (secret1_batch, secret2_batch, secret3_batch, cover_batch),
                                                         axis=3)))
                # reveal loss
                decode1_loss.append(decoder1_model.train_on_batch(x=cover_loss, y=secret1_batch))
                decode2_loss.append(decoder2_model.train_on_batch(x=cover_loss, y=secret2_batch))
                decode3_loss.append(decoder3_model.train_on_batch(x=cover_loss, y=secret3_batch))

                K.set_value(autoencoder_model.optimizer.lr, self.lr_schedule(epoch))
                K.set_value(decoder1_model.optimizer.lr, self.lr_schedule(epoch))
                K.set_value(decoder2_model.optimizer.lr, self.lr_schedule(epoch))
                K.set_value(decoder3_model.optimizer.lr, self.lr_schedule(epoch))

            if (epoch % (epochs / 5)) == 0:
                encoder_model.save_weights(path + "encoder " + str(epoch))
                decoder1_model.save_weights(path + "decoder1 " + str(epoch))
                decoder2_model.save_weights(path + "decoder2 " + str(epoch))
                decoder3_model.save_weights(path + "decoder3 " + str(epoch))
                autoencoder_model.save_weights(path + "autoencoder" + str(epoch))

            history.append(np.mean(auto_encode_loss))

        def save_files():

            with open(path + "autoencoder_loss.pckl", 'wb') as f:
                pickle.dump(auto_encode_loss, f)

            with open(path + "decode_loss1.pckl", 'wb') as f:
                pickle.dump(decode1_loss, f)

            with open(path + "decode_loss2.pckl", 'wb') as f:
                pickle.dump(decode2_loss, f)

            with open(path + "decode_loss3.pckl", 'wb') as f:
                pickle.dump(decode3_loss, f)

            with open(path + "loss_history.pckl", 'wb') as f:
                pickle.dump(history, f)

        save_files()

        return encoder_model, decoder1_model, decoder2_model, decoder3_model, autoencoder_model
