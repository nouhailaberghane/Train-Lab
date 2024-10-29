import pandas as pd
import tensorflow as tf

def load_data(data_file):
    # Charge les données à partir d'un fichier CSV
    data = pd.read_csv(data_file)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def create_model(layers_config):
    model = tf.keras.Sequential()
    for layer in layers_config:
        model.add(tf.keras.layers.Dense(units=layer['units'], activation=layer['activation']))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(data_file, layers_config, batch_size, epochs, learning_rate):
    X, y = load_data(data_file)
    model = create_model(layers_config)

    # Entraîne le modèle
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    
    return {
        'loss': history.history['loss'],
        'mae': history.history['mae'],
        'final_model': model
    }
