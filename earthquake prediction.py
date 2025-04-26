import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import traceback

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df)} rows")
        
        # Clean date columns
        df['Observed date and time'] = pd.to_datetime(df['Observed date and time'], errors='coerce')
        df = df.dropna(subset=['Observed date and time'])
        print(f"After dropping NaT: {len(df)} rows")
        
        # Encode epicenter locations
        le_epicenter = LabelEncoder()
        df['Epicenter_encoded'] = le_epicenter.fit_transform(df['Place Name of Epicenter'])
        
        # Normalize magnitude and seismic intensity
        scaler_magnitude = MinMaxScaler()
        scaler_intensity = MinMaxScaler()
        df['Magnitude_scaled'] = scaler_magnitude.fit_transform(df[['Magnitude']])
        df['Maximum Seismic Intensity'] = pd.to_numeric(df['Maximum Seismic Intensity'], errors='coerce').fillna(1)
        df['Intensity_scaled'] = scaler_intensity.fit_transform(df[['Maximum Seismic Intensity']])
        
        # Extract temporal features
        df['Day'] = df['Observed date and time'].dt.day
        df['Hour'] = df['Observed date and time'].dt.hour
        df['Month'] = df['Observed date and time'].dt.month
        
        # Get hour distribution for prediction
        hour_dist = df['Hour'].value_counts(normalize=True)
        print(f"Hour distribution: {hour_dist}")
        
        return df, le_epicenter, scaler_magnitude, scaler_intensity, hour_dist
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        traceback.print_exc()
        raise

# Create sequences for LSTM
def create_sequences(df, seq_length=10):  # Reduced sequence length
    try:
        features = ['Magnitude_scaled', 'Intensity_scaled', 'Epicenter_encoded', 'Day', 'Hour', 'Month']
        X, y_prob, y_mag, y_int, y_epi = [], [], [], [], []
        
        for i in range(len(df) - seq_length):
            seq = df[features].iloc[i:i+seq_length].values
            next_event = df.iloc[i+seq_length]
            X.append(seq)
            y_prob.append(1)  # Assume earthquake occurred
            y_mag.append(next_event['Magnitude_scaled'])
            y_int.append(next_event['Intensity_scaled'])
            y_epi.append(next_event['Epicenter_encoded'])
        
        X = np.array(X)
        print(f"Created {len(X)} sequences")
        return (X, 
                np.array(y_prob), 
                np.array(y_mag), 
                np.array(y_int), 
                np.array(y_epi))
    except Exception as e:
        print(f"Error in create_sequences: {str(e)}")
        traceback.print_exc()
        raise

# Build LSTM model
def build_lstm_model(seq_length, n_features, n_epicenters):
    try:
        inputs = Input(shape=(seq_length, n_features))
        lstm1 = LSTM(100, return_sequences=True)(inputs)
        lstm2 = LSTM(50)(lstm1)
        dropout = Dropout(0.2)(lstm2)
        
        # Output for earthquake probability
        prob_output = Dense(1, activation='sigmoid', name='prob')(dropout)
        
        # Output for magnitude
        mag_output = Dense(1, name='magnitude')(dropout)
        
        # Output for intensity
        int_output = Dense(1, name='intensity')(dropout)
        
        # Output for epicenter
        epi_output = Dense(n_epicenters, activation='softmax', name='epicenter')(dropout)
        
        model = Model(inputs=inputs, outputs=[prob_output, mag_output, int_output, epi_output])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss={'prob': 'binary_crossentropy', 
                           'magnitude': 'mse', 
                           'intensity': 'mse', 
                           'epicenter': 'sparse_categorical_crossentropy'},
                     metrics={'prob': 'accuracy', 
                              'magnitude': 'mae', 
                              'intensity': 'mae', 
                              'epicenter': 'accuracy'})
        return model
    except Exception as e:
        print(f"Error in build_lstm_model: {str(e)}")
        traceback.print_exc()
        raise

# Generate future predictions
def predict_future(model, last_sequence, le_epicenter, scaler_magnitude, scaler_intensity, hour_dist, days=365):
    try:
        predictions = []
        current_sequence = last_sequence.copy()
        current_date = datetime(2025, 4, 26)
        
        for _ in range(days):
            current_date += timedelta(days=1)
            X = current_sequence[np.newaxis, :, :]
            prob, mag, intens, epi = model.predict(X, verbose=0)
            
            if prob[0][0] > 0.1:  # Lowered threshold
                mag_pred = scaler_magnitude.inverse_transform(mag)[0][0]
                intens_pred = scaler_intensity.inverse_transform(intens)[0][0]
                epi_pred = le_epicenter.inverse_transform([np.argmax(epi[0])])[0]
                
                # Sample hour from historical distribution
                pred_hour = np.random.choice(hour_dist.index, p=hour_dist.values)
                pred_time = f"{int(pred_hour):02d}:00"
                
                predictions.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Time': pred_time,
                    'Epicenter': epi_pred,
                    'Magnitude': round(mag_pred, 1),
                    'Seismic Intensity': round(intens_pred, 1),
                    'Probability': round(prob[0][0], 3)
                })
            
            # Update sequence with predicted values
            new_row = np.zeros(current_sequence.shape[1])
            new_row[0] = mag[0][0] if prob[0][0] > 0.1 else 0
            new_row[1] = intens[0][0] if prob[0][0] > 0.1 else 0
            new_row[2] = np.argmax(epi[0]) if prob[0][0] > 0.1 else 0
            new_row[3] = current_date.day
            new_row[4] = pred_hour if prob[0][0] > 0.1 else 0
            new_row[5] = current_date.month
            current_sequence = np.vstack((current_sequence[1:], new_row))
        
        print(f"Generated {len(predictions)} predictions")
        return pd.DataFrame(predictions)
    except Exception as e:
        print(f"Error in predict_future: {str(e)}")
        traceback.print_exc()
        raise

# Visualize predictions with line graphs
def visualize_predictions(predictions, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if predictions.empty:
            print("No predictions to visualize")
            return None
        
        # Convert date to datetime and sort
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        predictions = predictions.sort_values('Date')
        
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predicted Earthquake Characteristics (Apr 2025 - Apr 2026)', fontsize=16)
        
        # 1. Magnitude Over Time
        axes[0, 0].plot(predictions['Date'], predictions['Magnitude'], marker='o', linestyle='-', color='blue', markersize=4)
        axes[0, 0].set_title('Magnitude Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].grid(True)
        
        # Rotate x-axis labels for better readability
        for tick in axes[0, 0].get_xticklabels():
            tick.set_rotation(45)
        
        # 2. Seismic Intensity Over Time
        axes[0, 1].plot(predictions['Date'], predictions['Seismic Intensity'], marker='o', linestyle='-', color='red', markersize=4)
        axes[0, 1].set_title('Seismic Intensity Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Seismic Intensity')
        axes[0, 1].grid(True)
        
        for tick in axes[0, 1].get_xticklabels():
            tick.set_rotation(45)
        
        # 3. Earthquake Probability Over Time
        axes[1, 0].plot(predictions['Date'], predictions['Probability'], marker='o', linestyle='-', color='green', markersize=4)
        axes[1, 0].set_title('Earthquake Probability Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].grid(True)
        
        for tick in axes[1, 0].get_xticklabels():
            tick.set_rotation(45)
        
        # 4. Top 5 Epicenter Locations Over Time
        top_epicenters = predictions['Epicenter'].value_counts().head(5).index
        for epicenter in top_epicenters:
            epicenter_data = predictions[predictions['Epicenter'] == epicenter]
            axes[1, 1].plot(epicenter_data['Date'], epicenter_data['Magnitude'], 
                            marker='o', linestyle='', label=epicenter, markersize=6)
        
        axes[1, 1].set_title('Top 5 Epicenter Locations (Magnitude)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(45)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(output_dir, 'earthquake_predictions_line_graphs.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Line graphs saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in visualize_predictions: {str(e)}")
        traceback.print_exc()
        raise

# Main execution
def main():
    try:
        # Load data
        file_path = 'D:\\Japan Earthquake prediction\\Japan earthquake dataset.csv'
        if not os.path.exists(file_path):
            print(f"Dataset file not found at {file_path}")
            return
        
        df, le_epicenter, scaler_magnitude, scaler_intensity, hour_dist = load_and_preprocess_data(file_path)
        
        # Create sequences
        seq_length = 10  # Reduced to handle small dataset
        X, y_prob, y_mag, y_int, y_epi = create_sequences(df, seq_length)
        if len(X) == 0:
            print("No sequences created. Check dataset size or sequence length.")
            return
        
        # Build and train model
        model = build_lstm_model(seq_length, X.shape[2], len(le_epicenter.classes_))
        model.fit(X, {'prob': y_prob, 'magnitude': y_mag, 'intensity': y_int, 'epicenter': y_epi},
                  epochs=50, batch_size=32, verbose=1, validation_split=0.2)
        
        # Predict future
        last_sequence = X[-1]
        future_predictions = predict_future(model, last_sequence, le_epicenter, scaler_magnitude, scaler_intensity, hour_dist)
        
        # Save predictions
        output_dir = 'D:\\Japan Earthquake prediction'
        output_file = os.path.join(output_dir, 'earthquake_predictions_2026.csv')
        future_predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Visualize predictions
        vis_path = visualize_predictions(future_predictions, output_dir)
        print(f"Line graph visualization saved to {vis_path}")
        
        # Display first 100 predictions
        print("\nFirst 100 Predictions (or all if fewer):")
        print(future_predictions.head(100).to_string(index=False))
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()