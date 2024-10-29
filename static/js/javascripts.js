document.getElementById('trainForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const dataFile = document.getElementById('dataFile').files[0];
    if (!dataFile) {
        alert('Veuillez sélectionner un fichier de données.');
        return;
    }
    
    const numLayers = parseInt(document.getElementById('numLayers').value, 10);
    const layersConfig = [];
    
    for (let i = 0; i < numLayers; i++) {
        const units = document.getElementById(`layer${i}_units`).value;
        const activation = document.getElementById(`layer${i}_activation`).value;
        layersConfig.push({ units: units, activation: activation });
    }
    
    const batchSize = document.getElementById('batchSize').value;
    const epochs = document.getElementById('epochs').value;

    fetch('/train', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            data_file: dataFile.name,
            num_layers: numLayers,
            layers_config: layersConfig,
            batch_size: batchSize,
            epochs: epochs,
            learning_rate: 0.001 // Par défaut
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erreur lors de l\'envoi des données');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('results').innerHTML = `
            <h3>Résultats de l'entraînement</h3>
            <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
    })
    .catch(error => {
        console.error('Erreur:', error);
        document.getElementById('results').innerHTML = `
            <h3>Erreur</h3>
            <p>${error.message}</p>
        `;
    });
});

document.getElementById('addLayer').addEventListener('click', function() {
    const numLayers = parseInt(document.getElementById('numLayers').value, 10);
    const layerDiv = document.createElement('div');
    layerDiv.className = 'form-group';
    layerDiv.innerHTML = `
        <label for="layer${numLayers}">Configuration de la Couche ${numLayers + 1}</label>
        <input type="number" class="form-control" id="layer${numLayers}_units" placeholder="Nombre d'unités" required>
        <select class="form-control" id="layer${numLayers}_activation">
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
        </select>
    `;
    document.getElementById('layersConfig').appendChild(layerDiv);
    document.getElementById('numLayers').value = numLayers + 1; // Incrémente le nombre de couches
});
