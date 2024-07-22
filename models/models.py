from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras import layers
from tensorflow.keras.models import Model
from keras_applications.imagenet_utils import _obtain_input_shape, decode_predictions
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Flatten, Input, LSTM, Dense, Dropout, BatchNormalization
from keras.applications import InceptionV3, ResNet50, VGG16


class VGG16:
    def __init__(self, input_shape, filter_size, kernel_size=3 ):
        self.features=None
        self.input_shape=input_shape
        self.filter_size=filter_size
        self.kernel_size=kernel_size
        self.model=self.build_models()


    def build_models(self):
        x_input = Input(shape=self.input_shape)
        x = Conv1D(self.filter_size, self.kernel_size, activation='relu', padding='same', name='block1_conv1')(x_input)
        x = Conv1D(self.filter_size, self.kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block1_pool')(x)

        # Block 2
        x = Conv1D(self.filter_size*2, self.kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv1D(self.filter_size*2, self.kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling1D(2, strides=2, name='block2_pool')(x)

        # Block 3
        x = Conv1D(self.filter_size*4, self.kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv1D(self.filter_size*4, self.kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv1D(self.filter_size*4, self.kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling1D(2, strides=2, name='block3_pool')(x)

        # Block 4
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling1D(2, strides=2, name='block4_pool')(x)

        # Block 5
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv1D(self.filter_size*8, self.kernel_size, activation='relu', padding='same', name='block5_conv3')(x)
        self.features = MaxPooling1D(2, strides=2, name='block5_pool')(x)

        # x = Flatten(name='flatten')(x)
        # x = Dense(dense_units, activation='relu', name='fc1')(x)
        # x = Dense(dense_units, activation='relu', name='fc2')(x)
        # x = Dense(1, activation='sigmoid', name='predictions')(x)  # Sigmoid for binary classification

        x = GlobalAveragePooling1D()(self.features)
        # x = GlobalMaxPooling1D()(x)
        x = Dense(1, activation='linear', name='predictions')(x)  # Sigmoid for binary classification

        # Create model.
        model = Model(x_input, x, name='vgg16')
        return model

class LSTM_model:
    def __init__(self, input_shape):
        self.features=None
        self.input_shape=input_shape
        self.model=self.build_models()



    def build_models(self):
        inputs = Input(shape=self.input_shape)
        x = LSTM(units=128)(inputs)  # Adjust units and other parameters as needed
        x = Dropout(0.4)(x)
        x = Dense(units=96, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(0.5)(x)
        self.features = Dense(units=32, activation='relu')(x)
        x = Dropout(0.5)(self.features)
        outputs = Dense(units=1, activation='linear')(x)
        model = Model(inputs, outputs, name='model2')
        return model


class GRU:
    def __init__(self, input_shape):
        self.features=None
        self.input_shape=input_shape
        self.model=self.build_models()
    
    def build_models(self):
        inputs = Input(shape=self.input_shape)
        x = GRU(units=128)(inputs) 
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(0.5)(x)
        self.features = Dense(units=32, activation='relu')(x)
        x = Dropout(0.5)(self.features)
        outputs = Dense(units=1, activation='linear')(x)
        model = Model(inputs, outputs, name='model2')
        return model



