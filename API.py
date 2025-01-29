
from flask import Flask, request , jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('car_pricing_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
 
        input_data = pd.DataFrame({
            'car_ID': [data.get('car_ID', 0)], 
            'symboling': [data.get('symboling', 0)], 
            'CarName': [data['car_name']],
            'fueltype': [data['fuel_type']],
            'aspiration': [data['aspiration']],
            'doornumber': [data['door_number']],
            'carbody': [data['car_body']],
            'drivewheel': [data['drive_wheel']],
            'enginelocation': [data['engine_location']],
            'wheelbase': [float(data['wheelbase'])],
            'carlength': [float(data['car_length'])],
            'carwidth': [float(data['car_width'])],
            'carheight': [float(data['car_height'])],
            'curbweight': [float(data['curb_weight'])],
            'enginetype': [data['engine_type']],
            'cylindernumber': [data['cylinder_number']],
            'enginesize': [float(data['engine_size'])],
            'fuelsystem': [data['fuel_system']],
            'boreratio': [float(data['bore_ratio'])],
            'stroke': [float(data['stroke'])],
            'compressionratio': [float(data['compression_ratio'])],
            'horsepower': [float(data['horsepower'])],
            'peakrpm': [float(data['peak_rpm'])],
            'citympg': [float(data['city_mpg'])],
            'highwaympg': [float(data['highway_mpg'])],
            'enginesize_bins': [data['enginesize_bins']]
        })
        
     
        print("Input Data Received:")
        print(input_data)
        

        prediction = model.predict(input_data)
        
        return jsonify({
            'status': 'success',
            'predicted_price': float(prediction[0])
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)