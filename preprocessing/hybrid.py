from tensorflow import keras

from ml_framework.model import base
from ml_framework.model import utils


class HybridCNN(base.ClassificationModel):

  def get_input_shape(self):
    h = self.hparams
    return (h.height, h.width, h.channels)

  @staticmethod
  def create_model1(input_shape, hparams):
    layer_filters = utils.get_model_layers(
        hparams.network_depth,
        hparams.num_filters,
        hparams.filter_increase_mode,
    )

    model = keras.Sequential([keras.layers.Input(input_shape)])
    for filters in layer_filters:
      model.add(keras.layers.Conv1D(
          filters, 3, padding='same'))
      if hparams.batchnorm == 1:
        model.add(keras.layers.BatchNormalization())
      if hparams.activation == 0:
        model.add(keras.layers.ReLU())
      elif hparams.activation == 1:
        model.add(keras.layers.LeakyReLU())
      else:
        raise NotImplementedError('Unsupported activation layer type.')
      if hparams.conv_dropout > 0:
        model.add(keras.layers.Dropout(hparams.conv_dropout))
      if hparams.downsampling == 0:  # MaxPool
        model.add(keras.layers.MaxPooling1D(2, padding='same'))
      elif hparams.downsampling == 1:  # Convolution downsample
        model.add(keras.layers.Conv1D(
            filters, 3, strides=2, padding='same'))
      else:
        raise NotImplementedError('Unsupported downsampling layer type.')

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        hparams.filter_multiplier * layer_filters[-1]))
    if hparams.dense_dropout > 0:
      model.add(keras.layers.Dropout(hparams.dense_dropout))
    if hparams.activation == 0:
      model.add(keras.layers.ReLU())
    elif hparams.activation == 1:
      model.add(keras.layers.LeakyReLU())
    else:
      raise NotImplementedError('Unsupported activation layer type.')

    return model

  @staticmethod
  def create_model2(input_shape, hparams):
    model = keras.Sequential([keras.layers.Input(input_shape)])
    for filters in layer_filters:
      model.add(keras.layers.Conv2D(filters, 3, padding='same'))
      if hparams.batchnorm == 1:
        model.add(keras.layers.BatchNormalization())
      if hparams.activation == 0:
        model.add(keras.layers.ReLU())
      elif hparams.activation == 1:
        model.add(keras.layers.LeakyReLU())
      else:
        raise NotImplementedError('Unsupported activation layer type.')
      if hparams.conv_dropout > 0:
        model.add(keras.layers.Dropout(hparams.conv_dropout))
      if hparams.downsampling == 0:  # MaxPool
        model.add(keras.layers.MaxPooling2D(2, padding='same'))
      elif hparams.downsampling == 1:  # Convolution downsample
        model.add(keras.layers.Conv2D(
            filters, 3, strides=2, padding='same',))
      else:
        raise NotImplementedError('Unsupported downsampling layer type.')

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        hparams.filter_multiplier * layer_filters[-1]))
    if hparams.dense_dropout > 0:
      model.add(keras.layers.Dropout(hparams.dense_dropout))
    if hparams.activation == 0:
      model.add(keras.layers.ReLU())
    elif hparams.activation == 1:
      model.add(keras.layers.LeakyReLU())
    else:
      raise NotImplementedError('Unsupported activation layer type.')

    return model

  @staticmethod
  def create_model(input_shape, hparams):
    input_shape1 = (input_shape[1], 6)
    model1 = create_model1(input_shape1, hparams)
    model2 = create_model2(input_shape, hparams)
    # combine the output of the two models
    combined = keras.concatenate([model1.output, model2.output])
    output = keras.layers.Dense(hparams.out_channels)(combined)
    return Model(inputs=[model1.input, model2.input], outputs=output)


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
