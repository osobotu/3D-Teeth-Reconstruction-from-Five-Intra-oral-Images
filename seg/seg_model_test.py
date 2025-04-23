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


def DenseASPP_ResNet(shape=(512, 512, 3), filters=[64, 128, 256, 512, 1024], kern_size=3):
    outputShape = shape[:2]
    conv_times = 2
    inp = layers.Input(shape)
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)
    skip_connections = [
        backbone.get_layer("conv1_relu").output,
        backbone.get_layer("conv2_block3_out").output,
        backbone.get_layer("conv3_block4_out").output,
        backbone.get_layer("conv4_block6_out").output,
    ]
    x = backbone.output  # Deepest feature map
    x = DenseASPP(x, filters[-1])  # Apply DenseASPP module

    # Decoder
    for i in reversed(range(len(skip_connections))):
        f = filters[i]
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=2, padding="valid")(x)
        x = layers.Concatenate()([x, skip_connections[i]])
        x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=0.1, dila=1)

    x = LeakyConv2D(x, filters=1, k_size=1, leaky_rate=0.1, dila=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name="DenseASPP-ResNet")
    return model

