import tensorflow as tf
from tensorflow import keras

def U_Net (o_w, o_h, o_c, t_w, t_h, t_c, num_layers, initial_ch, deep_sup):
    global model
    def encoder(e_in, filters, kernel_size=(3, 3), padding="same", strides=1):
        e = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(e_in)
        skip_out = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(e)
        e_out = keras.layers.MaxPool2D((2, 2), (2, 2))(skip_out)
        return e_out, skip_out

    def bridge(bridge_in, filters, kernel_size=(3, 3), padding="same", strides=1):
        b = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(bridge_in)
        bridge_out = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(b)
        return bridge_out

    def decoder(d_in, skip_in, filters, kernel_size=(3, 3), padding="same", strides=1):
        up_sampled = keras.layers.UpSampling2D((2, 2))(d_in)
        up_sampled_reshaped = tf.keras.layers.experimental.preprocessing.Resizing(int((skip_in.shape[1])*t_h/o_h), int((skip_in.shape[2])*t_w/o_w), interpolation="bilinear", name=None)(up_sampled)
        skip_in_reshaped = tf.keras.layers.experimental.preprocessing.Resizing(up_sampled_reshaped.shape[1], up_sampled_reshaped.shape[2], interpolation="bilinear", name=None)(skip_in)
        concatenated = keras.layers.Concatenate()([up_sampled_reshaped, skip_in_reshaped])
        d = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concatenated)
        d_out = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(d)
        return d_out

    inputs = keras.layers.Input((o_h, o_w, o_c))
    dict = {}

    num_layers_str = str(num_layers-1)
    num_layers_minus2_str = str(num_layers-3)

    e_out_end = "e_out"+num_layers_str
    d_out_end = "d_out"+num_layers_str
    d_out_end_minus2 = "d_out"+num_layers_minus2_str

    dict['e_out0'] = inputs
    for i in range(1,(num_layers)):
        dict['e_out%s'%i], dict['skip_out%s'%i] = encoder((dict['e_out%s'%(i-1)]), (0.5*initial_ch*(2**i)))


    bridge_out = bridge((dict[e_out_end]), (0.5*initial_ch*(2**num_layers)))

    dict['d_out0'] = bridge_out

    for n in range(1,(num_layers)):
        dict['d_out%s'%n] = decoder((dict['d_out%s'%(n-1)]), (dict['skip_out%s'%(num_layers-n)]), (0.5*initial_ch*(2**(num_layers-n))))

    outputs = keras.layers.Conv2D(t_c, (1, 1), padding="same", activation="sigmoid")(dict[d_out_end])

    if deep_sup == 1:
        ds = keras.layers.UpSampling2D((4, 4))(dict[d_out_end_minus2])
        ds = keras.layers.Conv2D(t_c, (1, 1), padding="same", activation="sigmoid")(ds)
        outputsconcat = tf.keras.layers.concatenate([outputs, ds], axis=1)
        model = keras.models.Model(inputs, outputsconcat)

    if deep_sup == 0:
        model = keras.models.Model(inputs, outputs)

    return model


def modelcreator(o_w, o_h, o_c, t_w, t_h, t_c, model_type, num_layers, initial_ch, deep_sup, gan_onoff=0):
    global current_model
    # For U-Net:
    if model_type == "U-Net":
        current_model = U_Net (o_w, o_h, o_c, t_w, t_h, t_c, num_layers, initial_ch, deep_sup)



    return current_model
