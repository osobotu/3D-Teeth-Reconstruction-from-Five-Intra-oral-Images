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

# def DenseASPP(x, filters, d_rates=[3, 6, 12, 18, 24], leaky_rate=0.1):
#     concat_feats = [x]
#     for i, rate in enumerate(d_rates):
#         x = layers.Conv2D(filters, kernel_size=3, padding='same', dilation_rate=rate, activation='linear')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(alpha=leaky_rate)(x)
#         concat_feats.append(x)
#         x = layers.concatenate(concat_feats)
#     return x

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

def DenseASPP(x, filters, d_rates=[3, 6, 12, 18, 24], leaky_rate=0.1):
    concat_feats = [x]
    for i, rate in enumerate(d_rates):
        x = layers.Conv2D(filters, kernel_size=3, padding='same', dilation_rate=rate, activation='linear')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=leaky_rate)(x)
        concat_feats.append(x)
        x = layers.concatenate(concat_feats)
    return x

def DenseASPP_ResNet(input_shape, aspp_filters=256):
    """
    A segmentation model using pretrained ResNet50 as encoder
    with DenseASPP module and decoder
    """
    outputShape = input_shape[:2]
    
    # Create the base ResNet50 model without top layer
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # We'll use these layers from ResNet50 as skip connections
    resnet_layers = [
        'conv1_relu',     # 1/2 resolution
        'conv2_block3_out',  # 1/4 resolution
        'conv3_block4_out',  # 1/8 resolution
        'conv4_block6_out',  # 1/16 resolution
    ]
    
    # Get the output of these layers
    skip_connections = [base_model.get_layer(layer_name).output for layer_name in resnet_layers]
    
    # Feature maps from the final block of ResNet50 (1/32 resolution)
    x = base_model.get_layer('conv5_block3_out').output
    
    # Apply DenseASPP module
    x = DenseASPP(x, aspp_filters)
    
    # Decoder path with skip connections
    # Start with 1/32 resolution from the encoder
    
    # Upsample to 1/16 resolution
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    # Connect with ResNet features at 1/16 resolution
    x = layers.concatenate([x, skip_connections[3]])
    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Upsample to 1/8 resolution
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    # Connect with ResNet features at 1/8 resolution
    x = layers.concatenate([x, skip_connections[2]])
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Upsample to 1/4 resolution
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    # Connect with ResNet features at 1/4 resolution
    x = layers.concatenate([x, skip_connections[1]])
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Upsample to 1/2 resolution
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    # Connect with ResNet features at 1/2 resolution
    x = layers.concatenate([x, skip_connections[0]])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Final upsampling to full resolution
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Output layer
    x = layers.Conv2D(1, 1, activation='linear')(x)
    x = layers.Reshape(outputShape)(x)
    
    # Create model using input from the base_model
    model = keras.Model(inputs=base_model.input, outputs=x, name="DenseASPP-PretrainedResNet")
    
    return model