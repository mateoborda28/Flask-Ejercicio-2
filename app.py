from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import predict_model

# Crear una instancia de Flask
app = Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return "Predicciones Precio"

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones.json'

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "best_model.pkl"
with open(model_path, 'rb') as model_file:
    dt2 = pickle.load(model_file)

# Cargar base de predicción en kaggle
prueba = pd.read_csv("prueba_APP.csv", sep = ";", header = 0, decimal = ",")


covariables = ['dominio', 'Tec', 'Avg. Session Length','Time on App',
                 'Time on Website', 'Length of Membership']
# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

    # Endpoint para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Recibir los datos en formato JSON
    # Crear DataFrame a partir de los datos de entrada
    user_data = pd.DataFrame([data])
    user_data.drop(columns=["Email","Address"], inplace=True)
    prueba1 = prueba.get(covariables)
    # Asegurar que las columnas del DataFrame "user_data" coincidan con las de "prueba"
    user_data.columns = prueba1.columns

  
    # Concatenar los datos del usuario con los datos del CSV "prueba"
    prueba2 = pd.concat([user_data, prueba1], axis=0)
    prueba2.index = range(prueba2.shape[0])

    # Realizar predicción
    df_test = prueba2.copy()
    predictions = predict_model(dt2, data=df_test)

    predictions = predictions["prediction_label"]
    prediction_label = predictions.iloc[0,]
    prediction_label=np.round(float(prediction_label),2)

        # Guardar predicción con ID en el archivo JSON
    prediction_result = {"Email": data["Email"], "prediction": prediction_label}
    save_prediction(prediction_result)

    return jsonify(prediction_result)

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=8000)