import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')


class VegetationHealthPredictor:
    def __init__(self):
        """Initialize the vegetation health prediction model with Random Forest"""
        self.geolocator = Nominatim(user_agent="vegetation_health_predictor")
        self.ndvi_data = None
        self.weather_data = None
        self.combined_data = None
        self.location_coords = None
        self.location_name = None
        self.validation_results = {}
        self.scaler = StandardScaler()
        self.rf_model = None
        self.feature_cols = None
        self.is_trained = False

    def __getstate__(self):
        """Return state to be pickled"""
        state = self.__dict__.copy()
        # Don't pickle the geolocator object
        del state['geolocator']
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state"""
        self.__dict__.update(state)
        # Re-initialize the geolocator object
        self.geolocator = Nominatim(user_agent="vegetation_health_predictor")

    def geocode_location(self, location_name):
        """Convert location name to coordinates"""
        try:
            location = self.geolocator.geocode(location_name)
            if location:
                self.location_coords = (location.latitude, location.longitude)
                self.location_name = location_name
                print(f"Location: {location_name}")
                print(f"Coordinates: {self.location_coords}")
                return self.location_coords
            else:
                raise ValueError(
                    f"Could not find coordinates for {location_name}")
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None

    def extract_ndvi_data(self, lat, lon, start_date, end_date, buffer_size=1000):
        """Extract NDVI data from Google Earth Engine"""
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(buffer_size)

            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .select('NDVI')

            def extract_ndvi(image):
                ndvi = image.select('NDVI').multiply(0.0001)
                ndvi_mean = ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=250,
                    maxPixels=1e9
                ).get('NDVI')

                return ee.Feature(None, {
                    'date': image.date().format('YYYY-MM-dd'),
                    'ndvi': ndvi_mean
                })

            ndvi_features = ndvi_collection.map(extract_ndvi)
            ndvi_list = ndvi_features.getInfo()['features']

            ndvi_data = []
            for feature in ndvi_list:
                props = feature['properties']
                if props['ndvi'] is not None:
                    ndvi_data.append({
                        'date': props['date'],
                        'ndvi': props['ndvi']
                    })

            self.ndvi_data = pd.DataFrame(ndvi_data)
            self.ndvi_data['date'] = pd.to_datetime(self.ndvi_data['date'])
            self.ndvi_data = self.ndvi_data.sort_values('date')

            print(f"✓ Extracted {len(self.ndvi_data)} NDVI observations")
            return self.ndvi_data

        except Exception as e:
            print(f"Error extracting NDVI data: {e}")
            return None

    def extract_weather_data(self, lat, lon, start_date, end_date):
        """Extract weather data from NASA POWER API"""
        try:
            start_date_str = start_date.replace('-', '')
            end_date_str = end_date.replace('-', '')

            base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            parameters = [
                'T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR',
                'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN'
            ]

            url = f"{base_url}?parameters={','.join(parameters)}&community=AG&longitude={
                                                    lon}&latitude={lat}&start={start_date_str}&end={end_date_str}&format=JSON"

            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            weather_records = []
            for param in parameters:
                param_data = data['properties']['parameter'][param]
                for date_str, value in param_data.items():
                    if value != -999:
                        weather_records.append({
                            'date': datetime.strptime(date_str, '%Y%m%d').date(),
                            'parameter': param,
                            'value': value
                        })

            weather_df = pd.DataFrame(weather_records)
            self.weather_data = weather_df.pivot(
                index='date', columns='parameter', values='value').reset_index()
            self.weather_data['date'] = pd.to_datetime(
                self.weather_data['date'])

            column_mapping = {
                'T2M': 'temperature_avg',
                'T2M_MAX': 'temperature_max',
                'T2M_MIN': 'temperature_min',
                'PRECTOTCORR': 'precipitation',
                'RH2M': 'humidity',
                'WS2M': 'wind_speed',
                'ALLSKY_SFC_SW_DWN': 'solar_radiation'
            }
            self.weather_data = self.weather_data.rename(
                columns=column_mapping)

            print(f"✓ Extracted weather data for {
                  len(self.weather_data)} days")
            return self.weather_data

        except Exception as e:
            print(f"Error extracting weather data: {e}")
            return None

    def combine_datasets(self):
        """Combine NDVI and weather data with feature engineering"""
        if self.ndvi_data is None or self.weather_data is None:
            print("Both NDVI and weather data are required")
            return None

        self.combined_data = pd.merge(
            self.ndvi_data, self.weather_data, on='date', how='inner')
        self.combined_data = self.combined_data.sort_values('date')

        # Feature engineering
        self.combined_data['temperature_range'] = (
            self.combined_data['temperature_max'] -
                self.combined_data['temperature_min']
        )

        # Moving averages
        for window in [3, 7, 14]:
            self.combined_data[f'precipitation_ma_{
                window}'] = self.combined_data['precipitation'].rolling(window=window, center=True).mean()
            self.combined_data[f'temperature_ma_{window}'] = self.combined_data['temperature_avg'].rolling(
                window=window, center=True).mean()

        # Lagged features
        for lag in [1, 3, 7]:
            self.combined_data[f'ndvi_lag_{
                lag}'] = self.combined_data['ndvi'].shift(lag)
            self.combined_data[f'precipitation_lag_{
                lag}'] = self.combined_data['precipitation'].shift(lag)
            self.combined_data[f'temperature_lag_{
                lag}'] = self.combined_data['temperature_avg'].shift(lag)

        # Additional features
        self.combined_data['gdd'] = np.maximum(
            0, self.combined_data['temperature_avg'] - 10)
        self.combined_data['precip_cumsum_7'] = self.combined_data['precipitation'].rolling(
            window=7).sum()
        self.combined_data['precip_cumsum_30'] = self.combined_data['precipitation'].rolling(
            window=30).sum()

        # Seasonal features
        self.combined_data['day_of_year'] = self.combined_data['date'].dt.dayofyear
        self.combined_data['month'] = self.combined_data['date'].dt.month
        self.combined_data['season'] = self.combined_data['month'].apply(
            lambda x: 1 if x in [12, 1, 2] else 2 if x in [
                3, 4, 5] else 3 if x in [6, 7, 8] else 4
        )

        print(f"✓ Combined dataset: {len(self.combined_data)} observations, {
              len(self.combined_data.columns)} features")
        return self.combined_data

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive accuracy metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            directional_accuracy = np.mean(
                direction_true == direction_pred) * 100
        else:
            directional_accuracy = 0

        return {
            'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MAPE': mape, 'Directional_Accuracy': directional_accuracy
        }

    def train_model(self, target_column='ndvi', validation=False):
        """Train Random Forest model with cross-validation"""
        if self.combined_data is None:
            print("❌ No data available for training")
            return None

        self.feature_cols = [col for col in self.combined_data.columns
                            if col not in ['date', target_column]]

        clean_data = self.combined_data.dropna()

        if len(clean_data) < 20:
            print("❌ Not enough clean data for training")
            return None

        X = clean_data[self.feature_cols]
        y = clean_data[target_column]

        if validation:
            self._validate_model(clean_data, target_column)

        # Train final model
        X_scaled = self.scaler.fit_transform(X)

        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=15,
            min_samples_split=5, min_samples_leaf=2
        )
        self.rf_model.fit(X_scaled, y)
        self.is_trained = True

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("✅ Model trained successfully!")
        print(f"\n📊 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return self.rf_model

    def _validate_model(self, clean_data, target_column='ndvi', n_splits=5):
        """Internal method for model validation"""
        X = clean_data[self.feature_cols]
        y = clean_data[target_column]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        print(f"🔄 Performing {n_splits}-fold cross-validation...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            rf = RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=15)
            rf.fit(X_train_scaled, y_train)

            predictions = rf.predict(X_test_scaled)
            metrics = self.calculate_metrics(y_test.values, predictions)
            metrics['fold'] = fold + 1
            fold_results.append(metrics)

            print(
                f"  Fold {fold + 1}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")

        # Calculate average metrics
        avg_metrics = {}
        for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Directional_Accuracy']:
            values = [result[metric]
                for result in fold_results if not np.isnan(result[metric])]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)

        self.validation_results = {
            'fold_results': fold_results,
            'average_metrics': avg_metrics
        }

        print(f"\n📈 Cross-Validation Results:")
        print(f"Average RMSE: {avg_metrics.get(
            'avg_RMSE', 'N/A'):.4f} ± {avg_metrics.get('std_RMSE', 'N/A'):.4f}")
        print(f"Average R²: {avg_metrics.get('avg_R2', 'N/A')              :.4f} ± {avg_metrics.get('std_R2', 'N/A'):.4f}")
        print(f"Average MAPE: {avg_metrics.get(
            'avg_MAPE', 'N/A'):.1f}% ± {avg_metrics.get('std_MAPE', 'N/A'):.1f}%")

    def predict(self, forecast_days=30):
        """Make predictions for future dates"""
        if not self.is_trained:
            print("❌ Model not trained yet. Run train_model() first.")
            return None

        try:
            recent_data = self.combined_data.tail(30).copy()
            last_date = self.combined_data['date'].max()
            predictions_list = []

            print(f"🔮 Generating predictions for next {forecast_days} days...")

            for day in range(1, forecast_days + 1):
                future_date = last_date + timedelta(days=day)

                recent_weather = recent_data[['temperature_avg', 'temperature_max', 'temperature_min',
                                            'precipitation', 'humidity', 'wind_speed', 'solar_radiation']].mean()

                day_of_year = future_date.dayofyear
                month = future_date.month
                season = 1 if month in [12, 1, 2] else 2 if month in [
                    3, 4, 5] else 3 if month in [6, 7, 8] else 4

                feature_dict = {}

                # Basic weather features
                for col in ['temperature_avg', 'temperature_max', 'temperature_min',
                           'precipitation', 'humidity', 'wind_speed', 'solar_radiation']:
                    feature_dict[col] = recent_weather[col]

                # Derived features
                feature_dict['temperature_range'] = recent_weather['temperature_max'] - \
                    recent_weather['temperature_min']
                feature_dict['gdd'] = max(
                    0, recent_weather['temperature_avg'] - 10)

                # Moving averages
                for window in [3, 7, 14]:
                    feature_dict[f'precipitation_ma_{
                        window}'] = recent_data['precipitation'].tail(window).mean()
                    feature_dict[f'temperature_ma_{
                        window}'] = recent_data['temperature_avg'].tail(window).mean()

                # Lagged features
                for lag in [1, 3, 7]:
                    if len(recent_data) >= lag:
                        feature_dict[f'ndvi_lag_{
                            lag}'] = recent_data['ndvi'].iloc[-lag]
                        feature_dict[f'precipitation_lag_{
                            lag}'] = recent_data['precipitation'].iloc[-lag]
                        feature_dict[f'temperature_lag_{
                            lag}'] = recent_data['temperature_avg'].iloc[-lag]

                # Cumulative precipitation
                feature_dict['precip_cumsum_7'] = recent_data['precipitation'].tail(
                    7).sum()
                feature_dict['precip_cumsum_30'] = recent_data['precipitation'].tail(
                    30).sum()

                # Seasonal features
                feature_dict['day_of_year'] = day_of_year
                feature_dict['month'] = month
                feature_dict['season'] = season

                # Create feature vector
                feature_vector = []
                for col in self.feature_cols:
                    if col in feature_dict:
                        feature_vector.append(feature_dict[col])
                    else:
                        feature_vector.append(self.combined_data[col].mean())

                # Make prediction
                feature_vector = np.array(feature_vector).reshape(1, -1)
                feature_vector_scaled = self.scaler.transform(feature_vector)
                predicted_ndvi = self.rf_model.predict(
                    feature_vector_scaled)[0]

                predictions_list.append({
                    'date': future_date,
                    'predicted_ndvi': predicted_ndvi
                })

                # Update recent_data for next iteration
                new_row = feature_dict.copy()
                new_row['date'] = future_date
                new_row['ndvi'] = predicted_ndvi
                recent_data = pd.concat(
                    [recent_data, pd.DataFrame([new_row])], ignore_index=True)
                recent_data = recent_data.tail(30)

            # Create predictions DataFrame
            predictions = pd.DataFrame(predictions_list)

            # Classify vegetation health
            predictions['health_status'] = predictions['predicted_ndvi'].apply(
                lambda x: 'Excellent' if x > 0.7 else
                         'Good' if x > 0.5 else
                         'Moderate' if x > 0.3 else
                         'Poor' if x > 0.1 else 'Very Poor'
            )

            print("✅ Predictions generated successfully!")
            return predictions

        except Exception as e:
            print(f"Error making predictions: {e}")
            return None

    def plot_results(self, predictions=None, figsize=(15, 10)):
        """Plot comprehensive results"""
        if self.combined_data is None:
            print("❌ No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: NDVI time series
        axes[0, 0].plot(self.combined_data['date'], self.combined_data['ndvi'],
                       label='Historical NDVI', color='green', alpha=0.7, linewidth=1)

        if predictions is not None:
            axes[0, 0].plot(predictions['date'], predictions['predicted_ndvi'],
                           label='Predicted NDVI', color='red', linestyle='--', linewidth=2, marker='o', markersize=3)

        axes[0, 0].set_title('NDVI Time Series and Predictions')
        axes[0, 0].set_ylabel('NDVI')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Weather variables
        axes[0, 1].plot(self.combined_data['date'], self.combined_data['temperature_avg'],
                       label='Temperature', color='orange', linewidth=2)
        ax2 = axes[0, 1].twinx()
        ax2.bar(self.combined_data['date'], self.combined_data['precipitation'],
                alpha=0.3, label='Precipitation', color='blue')
        axes[0, 1].set_title('Weather Variables')
        axes[0, 1].set_ylabel('Temperature (°C)')
        ax2.set_ylabel('Precipitation (mm)')
        axes[0, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: NDVI vs Temperature correlation
        axes[1, 0].scatter(self.combined_data['temperature_avg'],
                          self.combined_data['ndvi'], alpha=0.6, s=20)

        valid_temp = self.combined_data['temperature_avg'].dropna()
        valid_ndvi = self.combined_data['ndvi'].dropna()
        if len(valid_temp) > 1 and len(valid_ndvi) > 1:
            z = np.polyfit(valid_temp, valid_ndvi, 1)
            p = np.poly1d(z)
            temp_range = np.linspace(valid_temp.min(), valid_temp.max(), 100)
            axes[1, 0].plot(temp_range, p(temp_range), "r--", alpha=0.8)

        axes[1, 0].set_xlabel('Average Temperature (°C)')
        axes[1, 0].set_ylabel('NDVI')
        axes[1, 0].set_title('NDVI vs Temperature Correlation')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Health status distribution
        if predictions is not None:
            health_counts = predictions['health_status'].value_counts()
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
            axes[1, 1].pie(health_counts.values, labels=health_counts.index,
                          autopct='%1.1f%%', colors=colors[:len(health_counts)])
            axes[1, 1].set_title('Predicted Health Status Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Run predictions to see\nhealth status distribution',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.show()

    def get_summary(self, predictions=None):
        """Get a quick summary of the model and predictions"""
        if self.combined_data is None:
            return "❌ No data loaded"

        # Calculate NDVI statistics
        current_ndvi = self.combined_data['ndvi'].iloc[-1]
        min_ndvi = self.combined_data['ndvi'].min()
        max_ndvi = self.combined_data['ndvi'].max()
        avg_ndvi = self.combined_data['ndvi'].mean()

        summary = f"""
📍 Location: {self.location_name or 'Not specified'}
📅 Data Period: {self.combined_data['date'].min().strftime('%Y-%m-%d')} to {self.combined_data['date'].max().strftime('%Y-%m-%d')}
📊 Total Observations: {len(self.combined_data)}

📈 Current NDVI: {current_ndvi:.3f}
📊 Historical NDVI Range: {min_ndvi:.3f} (min) to {max_ndvi:.3f} (max)
📊 Average NDVI: {avg_ndvi:.3f}
"""

        if predictions is not None:
            pred_min = predictions['predicted_ndvi'].min()
            pred_max = predictions['predicted_ndvi'].max()
            avg_pred = predictions['predicted_ndvi'].mean()
            trend_slope = np.polyfit(range(len(predictions)), predictions['predicted_ndvi'], 1)[0]
            trend = "📈 Improving" if trend_slope > 0.001 else "📉 Declining" if trend_slope < -0.001 else "➡️ Stable"

            summary += f"""
🔮 Predictions: Next {len(predictions)} days
📊 Predicted NDVI Range: {pred_min:.3f} (min) to {pred_max:.3f} (max)
📊 Average Predicted NDVI: {avg_pred:.3f}
📈 Trend: {trend}
"""

        return summary

    def save_model(self, filename='vegetation_model.sav'):
        """Save the trained model"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Model saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    @staticmethod
    def load_model(filename='vegetation_model.sav'):
        """Load a saved model"""
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

def quick_setup(location_name, start_date="2000-01-01", end_date="2025-12-31"):
    """Quick setup function for Colab"""
    print(f"🚀 Setting up vegetation health predictor for {location_name}...")

    # Initialize predictor
    predictor = VegetationHealthPredictor()

    # Geocode location
    coords = predictor.geocode_location(location_name)
    if coords is None:
        return None

    lat, lon = coords

    print("\n📡 Extracting satellite and weather data...")

    # Extract NDVI data
    ndvi_data = predictor.extract_ndvi_data(lat, lon, start_date, end_date)
    if ndvi_data is None:
        print("❌ Failed to extract NDVI data")
        return None

    # Extract weather data
    weather_data = predictor.extract_weather_data(lat, lon, start_date, end_date)
    if weather_data is None:
        print("❌ Failed to extract weather data")
        return None

    # Combine datasets
    combined_data = predictor.combine_datasets()
    if combined_data is None:
        print("❌ Failed to combine datasets")
        return None

    print("✅ Data extraction complete!")
    return predictor

def train_and_predict(predictor, forecast_days=30):
    """Train model and make predictions in one go"""
    if predictor is None:
        print("❌ No predictor provided")
        return None

    print("🤖 Training model...")
    model = predictor.train_model(validation=False)
    if model is None:
        return None

    print("🔮 Making predictions...")
    predictions = predictor.predict(forecast_days=forecast_days)

    summary = predictor.get_summary(predictions)
    return summary

