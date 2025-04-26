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
Example Output:
First 100 Predictions (or all if fewer):
      Date  Time                                 Epicenter  Magnitude  Seismic Intensity  Probability
2025-04-27 04:00 Off the west Coast of Ishikawa Prefecture        2.8                1.9        0.965
2025-04-28 08:00                Northern Nagano Prefecture        2.0                1.3        0.978
2025-04-29 03:00                Northern Nagano Prefecture        2.5                1.2        0.979
2025-04-30 06:00 Off the west Coast of Ishikawa Prefecture        3.2                1.2        0.975
2025-05-01 20:00 Off the west Coast of Ishikawa Prefecture        2.9                1.8        0.966
2025-05-02 20:00 Off the west Coast of Ishikawa Prefecture        2.5                1.4        0.977
2025-05-03 22:00                Northern Nagano Prefecture        1.6                1.7        0.981
2025-05-04 15:00                Northern Nagano Prefecture        1.9                1.8        0.980
2025-05-05 06:00                Northern Nagano Prefecture        2.3                1.7        0.980
2025-05-06 03:00 Off the west Coast of Ishikawa Prefecture        2.5                1.5        0.977
2025-05-07 22:00   Off the east Coast of Aomori Prefecture        2.0                1.3        0.975
2025-05-08 00:00 Off the west Coast of Ishikawa Prefecture        2.8                1.6        0.973
2025-05-09 02:00   Off the east Coast of Aomori Prefecture        1.8                1.4        0.972
2025-05-10 20:00   Off the east Coast of Aomori Prefecture        1.3                1.2        0.965
2025-05-11 20:00   Off the east Coast of Aomori Prefecture        1.6                1.3        0.964
2025-05-12 20:00   Off the east Coast of Aomori Prefecture        1.9                1.4        0.968
2025-05-13 05:00   Off the east Coast of Aomori Prefecture        2.1                1.4        0.969
2025-05-14 16:00   Off the east Coast of Aomori Prefecture        1.8                1.2        0.970
2025-05-15 01:00                Aizu, Fukushima Prefecture        2.2                1.3        0.972
2025-05-16 05:00                Aizu, Fukushima Prefecture        1.8                1.3        0.974
2025-05-17 19:00                Aizu, Fukushima Prefecture        1.8                1.6        0.973
2025-05-18 18:00                Aizu, Fukushima Prefecture        2.3                1.9        0.974
2025-05-19 05:00                Aizu, Fukushima Prefecture        2.5                2.1        0.974
2025-05-20 04:00            Adjacent Sea of Tokara Islands        3.1                2.1        0.977
2025-05-21 21:00            Adjacent Sea of Tokara Islands        4.0                2.1        0.982
2025-05-22 11:00            Adjacent Sea of Tokara Islands        3.8                2.1        0.985
2025-05-23 20:00                   Central Oita Prefecture        3.7                2.1        0.984
2025-05-24 16:00                   Central Oita Prefecture        3.2                2.2        0.975
2025-05-25 06:00              Northern Wakayama Prefecture        3.9                3.2        0.924
2025-05-26 22:00              Northern Wakayama Prefecture        3.5                3.3        0.928
2025-05-27 19:00                Northern Nagano Prefecture        2.7                2.8        0.948
2025-05-28 20:00              Northern Wakayama Prefecture        3.6                3.1        0.934
2025-05-29 22:00              Northern Wakayama Prefecture        3.6                3.0        0.935
2025-05-30 08:00 Off the west Coast of Ishikawa Prefecture        3.4                2.8        0.953
2025-05-31 00:00                Northern Nagano Prefecture        2.4                1.8        0.976
2025-06-01 02:00 Off the west Coast of Ishikawa Prefecture        3.2                2.0        0.977
2025-06-02 05:00                Northern Nagano Prefecture        1.3                1.5        0.985
2025-06-03 18:00 Off the west Coast of Ishikawa Prefecture        1.7                1.3        0.986
2025-06-04 23:00 Off the west Coast of Ishikawa Prefecture        1.9                1.2        0.983
2025-06-05 01:00   Off the east Coast of Aomori Prefecture        1.8                1.3        0.979
2025-06-06 18:00 Off the west Coast of Ishikawa Prefecture        1.4                1.2        0.976
2025-06-07 21:00   Off the east Coast of Aomori Prefecture        1.8                1.5        0.975
2025-06-08 22:00                Aizu, Fukushima Prefecture        2.3                1.7        0.974
2025-06-09 14:00                Aizu, Fukushima Prefecture        2.8                1.9        0.975
2025-06-10 17:00                Aizu, Fukushima Prefecture        3.1                2.0        0.973
2025-06-11 23:00                Aizu, Fukushima Prefecture        2.8                2.4        0.974
2025-06-12 05:00                Aizu, Fukushima Prefecture        3.1                2.2        0.976
2025-06-13 16:00                Northern Nagano Prefecture        3.5                2.0        0.978
2025-06-14 16:00                Northern Nagano Prefecture        3.3                1.7        0.980
2025-06-15 20:00                Northern Nagano Prefecture        2.9                1.5        0.982
2025-06-16 00:00                Northern Nagano Prefecture        2.3                1.8        0.984
2025-06-17 11:00                Northern Nagano Prefecture        2.4                1.6        0.984
2025-06-18 08:00                Northern Nagano Prefecture        2.3                1.5        0.984
2025-06-19 14:00                Northern Nagano Prefecture        2.4                1.3        0.984
2025-06-20 06:00                Northern Nagano Prefecture        2.1                1.3        0.986
2025-06-21 13:00                Northern Nagano Prefecture        2.2                1.4        0.987
2025-06-22 18:00                Northern Nagano Prefecture        2.3                2.1        0.985
2025-06-23 20:00                Northern Nagano Prefecture        2.1                2.2        0.985
2025-06-24 22:00                Northern Nagano Prefecture        2.0                2.3        0.985
2025-06-25 05:00                Northern Nagano Prefecture        2.0                2.3        0.985
2025-06-26 04:00                Northern Nagano Prefecture        2.5                2.2        0.985
2025-06-27 17:00                   Central Oita Prefecture        2.8                2.1        0.985
2025-06-28 02:00                   Central Oita Prefecture        2.8                1.9        0.988
2025-06-29 04:00                   Central Oita Prefecture        2.7                1.7        0.987
2025-06-30 17:00                   Central Oita Prefecture        3.0                1.5        0.987
2025-07-01 16:00                   Central Oita Prefecture        2.8                1.8        0.984
2025-07-02 18:00                   Central Oita Prefecture        1.9                2.3        0.984
2025-07-03 11:00                   Central Oita Prefecture        2.3                1.9        0.982
2025-07-04 22:00              Northern Wakayama Prefecture        3.1                2.0        0.971
2025-07-05 10:00              Northern Wakayama Prefecture        3.6                2.0        0.971
2025-07-06 09:00 Off the west Coast of Ishikawa Prefecture        4.0                1.8        0.978
2025-07-07 20:00 Off the west Coast of Ishikawa Prefecture        3.9                1.4        0.977
2025-07-08 22:00 Off the west Coast of Ishikawa Prefecture        3.8                1.3        0.976
2025-07-09 02:00   Off the east Coast of Aomori Prefecture        2.1                1.1        0.976
2025-07-10 03:00                Northern Nagano Prefecture        1.9                0.8        0.971
2025-07-11 03:00                Northern Nagano Prefecture        2.1                0.9        0.970
2025-07-12 09:00                Northern Nagano Prefecture        1.9                0.7        0.968
2025-07-13 22:00   Off the east Coast of Aomori Prefecture        2.1                1.0        0.971
2025-07-14 04:00   Off the east Coast of Aomori Prefecture        2.5                1.4        0.973
2025-07-15 18:00   Off the east Coast of Aomori Prefecture        2.1                1.2        0.974
2025-07-16 15:00                Aizu, Fukushima Prefecture        2.4                1.4        0.975
2025-07-17 19:00                Aizu, Fukushima Prefecture        2.2                1.7        0.976
2025-07-18 05:00                Aizu, Fukushima Prefecture        2.1                2.0        0.975
2025-07-19 22:00            Adjacent Sea of Tokara Islands        2.0                2.1        0.974
2025-07-20 05:00            Adjacent Sea of Tokara Islands        2.1                2.0        0.977
2025-07-21 05:00            Adjacent Sea of Tokara Islands        2.9                2.1        0.977
2025-07-22 03:00                   Central Oita Prefecture        3.9                1.9        0.982
2025-07-23 19:00                   Central Oita Prefecture        4.0                1.6        0.986
2025-07-24 07:00                   Central Oita Prefecture        3.5                1.4        0.985
2025-07-25 21:00                   Central Oita Prefecture        3.0                1.9        0.975
2025-07-26 18:00                   Central Oita Prefecture        3.5                2.8        0.932
2025-07-27 03:00              Northern Wakayama Prefecture        3.9                2.8        0.925
2025-07-28 09:00              Northern Wakayama Prefecture        4.3                2.7        0.919
2025-07-29 06:00              Northern Wakayama Prefecture        4.3                2.8        0.919
2025-07-30 09:00              Northern Wakayama Prefecture        4.5                2.6        0.916
2025-07-31 21:00              Northern Wakayama Prefecture        4.4                2.6        0.918
2025-08-01 16:00              Northern Wakayama Prefecture        4.2                2.6        0.927
2025-08-02 01:00              Northern Wakayama Prefecture        3.0                2.5        0.941
2025-08-03 22:00              Northern Wakayama Prefecture        3.0                2.2        0.954
2025-08-04 18:00 Off the west Coast of Ishikawa Prefecture        3.8                1.8        0.977


