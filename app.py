from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from data_cleaning import DataCleaning, LinearRegressionModel, PredictionModel, RegressionPlot
import os
import pickle
import traceback

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files.get('uploadedFile1')
    file2 = request.files.get('uploadedFile2')

    if not file1 or file1.filename == '':
        return 'First file is missing', 400
    if not file2 or file2.filename == '':
        return 'Second file is missing', 400

    filepath1 = os.path.join(upload_folder, file1.filename)
    file1.save(filepath1)
    filepath2 = os.path.join(upload_folder, file2.filename)
    file2.save(filepath2)

    session['file1'] = filepath1
    session['file2'] = filepath2

    return redirect(url_for('index'))

@app.route('/train', methods=['GET'])
def train():
    if 'file1' not in session or 'file2' not in session:
        return redirect(url_for('index'))

    try:
        data_cleaner = DataCleaning(session['file1'], session['file2'])
        data_cleaner.run_all()

        modelo = LinearRegressionModel(data_cleaner.X, data_cleaner.y)
        modelo.split_data()
        modelo.train_model()
        modelo.make_predictions()
        mse, r2, rmse = modelo.evaluate_model()

        plotter = RegressionPlot(modelo.X_test, modelo.y_test, modelo.model, mse, r2, rmse)
        plot_div = plotter.plot_regression()

        # Guardar el modelo entrenado en el sistema de archivos
        model_path = os.path.join(upload_folder, 'modelo_entrenado.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(modelo.model, file)

        # Guardar los datos limpios en el sistema de archivos
        data_path = os.path.join(upload_folder, 'datos_limpios.pkl')
        with open(data_path, 'wb') as file:
            pickle.dump((data_cleaner.X, data_cleaner.y), file)

        # Almacenar las rutas en la sesión
        session['model_path'] = model_path
        session['data_path'] = data_path

        return render_template('index.html', plot_div=plot_div)
    except Exception as e:
        return f"An error occurred: {e}", 500

@app.route('/predict', methods=['GET'])
def predict():
    if 'model_path' not in session or 'data_path' not in session:
        return redirect(url_for('index'))

    try:
        # Cargar el modelo entrenado desde el sistema de archivos
        with open(session['model_path'], 'rb') as file:
            modelo_entrenado = pickle.load(file)

        # Cargar los datos limpios desde el sistema de archivos
        with open(session['data_path'], 'rb') as file:
            X, y = pickle.load(file)

        prediction_model = PredictionModel(modelo_entrenado, X)
        prediction_model.generate_random_data()
        prediction_model.make_predictions()
        graph_json = prediction_model.plot_predictions_line()

        return jsonify(graph_json)
    except Exception as e:
        app.logger.error('Error during prediction: %s', str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
