
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras import backend as K
from src.global_config import INPUT_SHAPE

input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format
n_features = 64
x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(n_features, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(n_features, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
flattened = Flatten()(encoded)

print(dir(flattened))
print(encoded.shape)
reshaped_2d = Reshape((10, 10, n_features))(flattened)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = BatchNormalization()(reshaped_2d)
x = Conv2D(n_features, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(n_features, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
Encoder = Model(input_img, flattened)

autoencoder.summary()