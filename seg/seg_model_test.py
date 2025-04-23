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


# def DenseASPP_ResNet(shape=(512, 512, 3), filters=[64, 128, 256, 512, 1024], kern_size=3):
#     outputShape = shape[:2]
#     conv_times = 2
#     inp = layers.Input(shape)
#     backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inp)
#     skip_connections = [
#         backbone.get_layer("conv1_relu").output,
#         backbone.get_layer("conv2_block3_out").output,
#         backbone.get_layer("conv3_block4_out").output,
#         backbone.get_layer("conv4_block6_out").output,
#     ]
#     x = backbone.output  # Deepest feature map
#     x = DenseASPP(x, filters[-1])  # Apply DenseASPP module

#     # Decoder
#     for i in reversed(range(len(skip_connections))):
#         f = filters[i]
#         x = layers.Conv2DTranspose(f, kernel_size=2, strides=2, padding="valid")(x)
#         x = layers.Concatenate()([x, skip_connections[i]])
#         x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=0.1, dila=1)

#     x = LeakyConv2D(x, filters=1, k_size=1, leaky_rate=0.1, dila=1)
#     print(x.shape)
#     # x = layers.Reshape(outputShape)(x)
#     model = keras.Model(inp, x, name="DenseASPP-ResNet")
#     return model

def ResNetBlock(x, filters, kernel_size=3, strides=1, leaky_rate=0.1):
    """
    A standard ResNet block with two convolutions and a skip connection
    """
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leaky_rate)(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection with projection if needed
    if strides > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(alpha=leaky_rate)(x)
    
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

def DenseASPP_ResNet(shape, filters=[64, 128, 256, 512], aspp_filters=256):
    """
    A ResNet-based encoder with DenseASPP module and decoder for segmentation
    """
    outputShape = shape[:2]
    input_tensor = layers.Input(shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # First skip connection before pooling
    skip0 = x  # This will be at 1/2 resolution
    
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Store encoder outputs for skip connections
    skip_connections = []
    skip_connections.append(x)  # This will be at 1/4 resolution
    
    # Track current resolution level (1/4 after initial conv + pooling)
    current_scale = 4
    scales = [current_scale]
    
    # ResNet encoder blocks with explicit dimension tracking
    for i, f in enumerate(filters):
        if i > 0:  # Downsample after first block
            x = ResNetBlock(x, f, strides=2)
            current_scale *= 2
        else:  # No downsampling in first block
            x = ResNetBlock(x, f, strides=1)
            
        x = ResNetBlock(x, f, strides=1)  # Additional processing at same resolution
        
        if i < len(filters) - 1:  # Don't store the last encoder output
            skip_connections.append(x)
            scales.append(current_scale)
    
    # Apply DenseASPP module
    x = DenseASPP(x, aspp_filters)
    
    # Decoder with careful upsampling to match skip connections
    for i in range(len(skip_connections) - 1, -1, -1):
        # Calculate upsampling factor based on current scale vs target scale
        x = layers.Conv2DTranspose(filters[min(i, len(filters)-1)], 2, strides=2, padding='same')(x)
        
        # Now concatenate with the proper skip connection
        x = layers.concatenate([x, skip_connections[i]])
        
        # Refine features
        x = ResNetBlock(x, filters[min(i, len(filters)-1)])
    
    # Final upsampling to match input resolution (from 1/4 to 1/2)
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x)
    
    # Concatenate with the earliest skip connection (before pooling)
    x = layers.concatenate([x, skip0])
    x = ResNetBlock(x, 64)
    
    # Final upsampling from 1/2 to full resolution
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Output layer
    x = layers.Conv2D(1, 1, activation='linear')(x)
    x = layers.Reshape(outputShape)(x)
    
    # Create model
    model = keras.Model(input_tensor, x, name="DenseASPP-ResNet")
    
    return model