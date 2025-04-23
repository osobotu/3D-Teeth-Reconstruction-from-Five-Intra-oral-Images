from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

#########################
#### network modules ####
#########################

def LeakyConv2D(x, filters, k_size=3, leaky_rate=0.1, dila=1):
    x = layers.Conv2D(filters, kernel_size=k_size, dilation_rate=dila, padding="same")(
        x
    )
    x = layers.LeakyReLU(leaky_rate)(x)
    return x


def CascadeConv2D(x, filters, conv_times, k_size=3, leaky_rate=0.1, dila=1):
    for _ in range(conv_times):
        x = LeakyConv2D(x, filters, k_size, leaky_rate, dila)
    return x

def DenseASPP(x, filters, d_rates=[3, 6, 12, 18, 24], leaky_rate=0.1):
    concat_feats = [x]
    for i, rate in enumerate(d_rates):
        x = layers.Conv2D(filters, kernel_size=3, padding='same', dilation_rate=rate, activation='linear')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=leaky_rate)(x)
        concat_feats.append(x)
        x = layers.concatenate(concat_feats)
    return x

def SeparableConv2D(x, filters, dila=1, leaky_rate=0.1):
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        dilation_rate=dila,
        use_bias=False,
    )(x)
    x = layers.LeakyReLU(leaky_rate)(x)
    x = layers.Conv2D(
        filters, kernel_size=(1, 1), strides=1, padding="same", use_bias=False
    )(x)
    x = layers.LeakyReLU(leaky_rate)(x)
    return x

def DenseASPP_UNet(shape, kern_size=3, filters=[64, 128, 256, 512, 1024]):
    outputShape = shape[:2]
    inp = layers.Input(shape)
    x = inp
    skips = []

    # Encoder
    for f in filters[:-1]:
        x = CascadeConv2D(x, f, conv_times=2, k_size=kern_size)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck with DenseASPP
    x = CascadeConv2D(x, filters[-1], conv_times=2, k_size=kern_size)
    x = DenseASPP(x, filters[-1])

    # Decoder
    for i in reversed(range(len(skips))):
        x = layers.Conv2DTranspose(filters[i], kernel_size=2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skips[i]])
        x = CascadeConv2D(x, filters[i], conv_times=2, k_size=kern_size)

    x = LeakyConv2D(x, filters=1, k_size=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name="DenseASPP-UNet")
    return model

def DenseASPP_ResNet(shape=(512, 512, 3), filters=256):
    backbone = ResNet50(include_top=False, weights="imagenet", input_shape=shape)
    skip_connections = [backbone.get_layer(layer_name).output for layer_name in 
                        ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']]

    x = backbone.output  # Deepest feature map
    x = DenseASPP(x, filters)

    # Decoder
    for skip in reversed(skip_connections):
        x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = CascadeConv2D(x, filters, conv_times=2)

    x = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(x)
    x = layers.Reshape(shape[:2])(x)
    model = keras.Model(inputs=backbone.input, outputs=x, name="DenseASPP-ResNet")
    return model
