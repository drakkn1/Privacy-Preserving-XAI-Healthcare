import tensorflow as tf

# -------------------------------------------------------
# MLP model used in the paper
# -------------------------------------------------------

def create_mlp(input_dim):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


# -------------------------------------------------------
# Simplified federated-style training
# (demonstration implementation)
# -------------------------------------------------------

def train_federated_model(X_train, y_train):

    input_dim = X_train.shape[1]

    model = create_mlp(input_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    # Simulated training (stand-in for federated rounds)
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    return model