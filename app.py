import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, send_file, url_for
import cv2
import os
app = Flask(__name__)

@app.route('/')
def index0():
    return render_template('index0.html')  # Assure-toi d'avoir un fichier index.html dans le dossier templates

@app.route('/train', methods=['POST'])
def train():
    # Charger le fichier CSV
    file = request.files['data-file']
    if file:
        df = pd.read_csv(file)

        # Conserver uniquement les colonnes numériques
        df = df.select_dtypes(include=[np.number])

        # Vérifier qu'il reste des colonnes après suppression des colonnes non numériques
        if df.empty:
            return "Erreur: Aucun des champs n'est numérique."

        # Définir X et y
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Configurer le modèle TensorFlow
        layer_types = request.form.getlist('layer-type[]')  # Liste des types de couches
        units = request.form.getlist('units[]')  # Liste des unités

        learning_rate = float(request.form['learning-rate'])

        # Créer le modèle
        model = tf.keras.Sequential()
        for layer_type, unit in zip(layer_types, units):
            unit = int(unit)  # Convertir les unités en entier
            if layer_type == 'Dense':
                model.add(tf.keras.layers.Dense(unit, activation='relu'))
            elif layer_type == 'Convolutional':
                model.add(tf.keras.layers.Conv2D(unit, (3, 3), activation='relu'))

        model.add(tf.keras.layers.Dense(1))  # Couche de sortie
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse',
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        # Entraîner le modèle
        history = model.fit(X, y, epochs=10, validation_split=0.2)

        # Générer et enregistrer la courbe de perte et de précision
        plt.figure(figsize=(12, 5))

        # Courbe de perte
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Perte d\'entraînement')
        plt.plot(history.history['val_loss'], label='Perte de validation')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.title('Courbe de perte')
        plt.legend()

        # Courbe de précision (MAE)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mean_absolute_error'], label='Précision d\'entraînement')
        plt.plot(history.history['val_mean_absolute_error'], label='Précision de validation')
        plt.xlabel('Époques')
        plt.ylabel('Précision')
        plt.title('Courbe de précision')
        plt.legend()

        # Enregistrer l'image avec les deux courbes
        plt.tight_layout()
        plt.savefig('static/plot.png')  # Enregistrer l'image dans le dossier static
        plt.close()  # Fermer la figure

        # Enregistrer le modèle
        model.save('model.h5')

        # Rediriger vers la page de visualisation
        return redirect(url_for('visualization'))

@app.route('/download')
def download():
    return send_file('model.h5', as_attachment=True)

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')
@app.route('/visualization_class_num')
def visualization_class_num():
    return render_template('visualization_class_num.html')

@app.route('/visualization_img')
def visualization_img():
    return render_template('visualization_img.html')



@app.route('/choose_model', methods=['POST'])
def choose_model():
    model_type = request.form['model_type']
    if model_type == 'regression':
        return redirect(url_for('regression_form'))
    elif model_type == 'classification':
        return redirect(url_for('classification_form'))

@app.route('/regression_form')
def regression_form():
    return render_template('regression.html')  # Your regression form page

@app.route('/classification_form')
def classification_form():
    return render_template('classification.html')

@app.route('/train_classification', methods=['POST'])
def train_classification():
    # Charger le fichier CSV
    file = request.files['data-file']
    if file:
        df = pd.read_csv(file)

        # Conserver uniquement les colonnes numériques
        df = df.select_dtypes(include=[np.number])

        # Vérifier qu'il reste des colonnes après suppression des colonnes non numériques
        if df.empty:
            return "Erreur: Aucun des champs n'est numérique."

        # Définir X et y
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Récupérer les paramètres du formulaire
        layer_units = request.form.getlist('units[]')
        hidden_activation = request.form['activation-function']
        output_activation = request.form['output-activation']
        learning_rate = float(request.form['learning-rate'])
        num_classes = int(request.form['num-classes'])

        # Créer le modèle
        model = tf.keras.Sequential()
        for units in layer_units:
            units = int(units)  # Convertir les unités en entier
            model.add(tf.keras.layers.Dense(units, activation=hidden_activation))

        # Définir la couche de sortie et la fonction de perte en fonction du nombre de classes
        if num_classes == 2:
            # Cas binaire : 1 unité de sortie avec 'sigmoid'
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            # Cas multiclasse : `num_classes` unités de sortie avec 'softmax'
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            y = tf.keras.utils.to_categorical(y, num_classes)
            loss = 'categorical_crossentropy'

        # Configurer le modèle
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss,
                      metrics=['accuracy'])

        # Entraîner le modèle
        history = model.fit(X, y, epochs=10, validation_split=0.2)

        # Générer et enregistrer la courbe de perte et de précision
        plt.figure(figsize=(12, 5))

        # Courbe de perte
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Perte d\'entraînement')
        plt.plot(history.history['val_loss'], label='Perte de validation')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.title('Courbe de perte')
        plt.legend()

        # Courbe de précision
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
        plt.plot(history.history['val_accuracy'], label='Précision de validation')
        plt.xlabel('Époques')
        plt.ylabel('Précision')
        plt.title('Courbe de précision')
        plt.legend()

        # Enregistrer l'image avec les deux courbes
        plt.tight_layout()
        plt.savefig('static/plot_classification.png')
        plt.close()

        # Enregistrer le modèle
        model.save('model_classification.h5')

        # Rediriger vers la page de visualisation
        return redirect(url_for('visualization_class_num'))
    





os.makedirs('uploads', exist_ok=True)



@app.route('/train_image', methods=['POST'])
def train_image():
    # Décoder et sauvegarder les images
    photos = request.form.getlist('photos[]')
    labels = []  # Ajouter la logique pour définir les labels si nécessaire
    for i, photo_data in enumerate(photos):
        # Supprimer le préfixe de la chaîne base64
        img_data = base64.b64decode(photo_data.split(",")[1])
        img_path = os.path.join('uploads', f'image_{i}.png')
        with open(img_path, 'wb') as f:
            f.write(img_data)

    # Charger les images pour l'entraînement
    images, labels = load_images_from_folder('uploads')  # Assurez-vous que cette fonction est définie

    # Normaliser les pixels entre 0 et 1
    images = images.astype('float32') / 255.0

    # Configurer le modèle avec les paramètres du formulaire
    layer_types = request.form.getlist('layer-type[]')
    units = request.form.getlist('units[]')
    learning_rate = float(request.form['learning-rate'])

    model = tf.keras.Sequential()
    for layer_type, unit in zip(layer_types, units):
        unit = int(unit)
        if layer_type == 'Convolutional':
            model.add(tf.keras.layers.Conv2D(unit, (3, 3), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Sortie pour classification binaire

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entraîner le modèle
    history = model.fit(images, labels, epochs=10, validation_split=0.2)

    # Générer les courbes de perte et de précision
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('Courbe de perte')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision de validation')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.legend()
    plt.title('Courbe de précision')

    plt.tight_layout()
    plt.savefig('static/plot_image.png')
    plt.close()

    model.save('model_image.h5')
    return redirect(url_for('visualization_img'))

# Assurez-vous de définir `load_images_from_folder`
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Ajustez la taille si nécessaire
            images.append(img)
            # Assigner un label selon le nom du fichier ou d'un critère
            label = 0 if 'class1' in filename else 1
            labels.append(label)
    return np.array(images), np.array(labels)

@app.route('/train_image')
def train_image_form():
    return render_template('train_image.html')

# Fonction de visualisation (à définir selon votre implémentation)
@app.route('/visualization_image')
def visualization_image():
    return render_template('visualization.html', plot_url='/static/plot_image.png')
@app.route('/index')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
