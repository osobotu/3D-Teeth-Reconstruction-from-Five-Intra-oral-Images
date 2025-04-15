from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def attention_gate(x, g, filters):
    theta_x = layers.Conv2D(filters, 1)(x)
    phi_g = layers.Conv2D(filters, 1)(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.ReLU()(add)
    psi = layers.Conv2D(1, 1, activation="sigmoid")(act)
    return layers.Multiply()([x, psi])

def ResUNetPlusPlus(shape=(512, 512, 3)):
    inputs = layers.Input(shape)
    
    # Encoder (ResNet blocks)
    e1 = residual_block(inputs, 64)
    e2 = residual_block(layers.MaxPooling2D(2)(e1), 128)
    e3 = residual_block(layers.MaxPooling2D(2)(e2), 256)
    
    # Bridge
    bridge = residual_block(layers.MaxPooling2D(2)(e3), 512)
    
    # Decoder with attention
    d3 = layers.UpSampling2D(2)(bridge)
    att3 = attention_gate(e3, d3, 256)
    d3 = layers.Concatenate()([d3, att3])
    d3 = residual_block(d3, 256)
    
    d2 = layers.UpSampling2D(2)(d3)
    att2 = attention_gate(e2, d2, 128)
    d2 = layers.Concatenate()([d2, att2])
    d2 = residual_block(d2, 128)
    
    d1 = layers.UpSampling2D(2)(d2)
    att1 = attention_gate(e1, d1, 64)
    d1 = layers.Concatenate()([d1, att1])
    d1 = residual_block(d1, 64)
    
    # Output (1 channel for binary segmentation)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    
    return keras.Model(inputs, outputs, name="ResUNetPlusPlus")