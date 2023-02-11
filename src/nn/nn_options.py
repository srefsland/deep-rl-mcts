import tensorflow as tf

activation_functions = {
    "relu": tf.keras.activations.relu,
    "sigmoid": tf.keras.activations.sigmoid,
    "tanh": tf.keras.activations.tanh,
    "softmax": tf.keras.activations.softmax,
    "linear": tf.keras.activations.linear
}

optimizers = {
    "Adagrad": tf.keras.optimizers.Adagrad,
    "SGD": tf.keras.optimizers.SGD,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "Adam": tf.keras.optimizers.Adam,
}

loss_functions = {
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy,
    "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy,
    "kl_divergence": tf.keras.losses.KLDivergence,
}
