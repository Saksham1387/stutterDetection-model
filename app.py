
import tempfile
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
CORS(app, supports_credentials=True, allow_headers=['Form-data', 'Authorization'], methods=['GET', 'POST', 'OPTIONS'])


with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the model
with open('rf_model_weights.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        Parth=[]

        y1, sr1 = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)
        spectral_centroid = librosa.feature.spectral_centroid(y=y1, sr=sr1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y1)
        duration = librosa.get_duration(y=y1, sr=sr1)
        duration = [duration]
        amplitude = np.max(np.abs(y1))
        amplitude = [amplitude]

        combined_features = np.concatenate([mfccs.mean(axis=1), spectral_bandwidth.mean(axis=1), spectral_centroid.mean(axis=1),
                                               zero_crossing_rate.mean(axis=1), duration, amplitude])
        Parth.append(combined_features)
        Parth = np.array(Parth)


        Parth_test = scaler.transform(Parth)
        y_pred=model.predict(Parth_test) 
        # return jsonify({'predicted_class': y_pred})
        prediction = y_pred.tolist()  # Convert NumPy array to list for JSON serialization
        response = jsonify({'predicted_class': prediction})
    except Exception as e:
        response = jsonify({'error': str(e)}), 500
    finally:
        os.remove(audio_path)  # Ensure the temporary file is deleted

    return response
if __name__ == '__main__':
    app.run(debug=True)