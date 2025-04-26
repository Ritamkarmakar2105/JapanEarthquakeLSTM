# JapanEarthquakeLSTM

Overview

JapanEarthquakeLSTM is a machine learning project that predicts earthquake characteristics in Japan, including date, time, epicenter, magnitude, and seismic intensity, for one year into the future (April 27, 2025, to April 26, 2026). The model uses a Long Short-Term Memory (LSTM) neural network trained on historical earthquake data. Predictions are visualized using line graphs in a 2x2 grid, showing magnitude, seismic intensity, probability, and top epicenter locations over time.
The project is designed for researchers, data scientists, and seismologists interested in earthquake forecasting and time-series analysis.
Features

LSTM Model: Multi-output LSTM predicts earthquake probability, magnitude, intensity, and epicenter.
Time Prediction: Specific dates and hours sampled from historical data distribution.
Visualization: Line graphs for:
Magnitude over time
Seismic intensity over time
Earthquake probability over time
Magnitude of top 5 epicenters (scatter plot)
Output: Predictions saved as a CSV and visualizations as a PNG.

Installation
Clone the Repository:
git clone https://github.com/<your-username>/JapanEarthquakeLSTM.git
cd JapanEarthquakeLSTM
Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
pip install -r requirements.txt
Verify Dataset:
update the file_path in earthquake_prediction_lstm_line_graphs.py to match your dataset location.
Usage
Run the Script:
python earthquake_prediction_lstm_line_graphs.py
Outputs:
Predictions: D:\Japan Earthquake prediction\earthquake_predictions_2026.csv
Columns: Date, Time, Epicenter, Magnitude, Seismic Intensity, Probability
Visualization:Japan Earthquake prediction\earthquake_predictions_line_graphs.png
2x2 grid of line graphs for magnitude, intensity, probability, and top epicenters.
Console: Displays debug messages and the first 100 predictions.


