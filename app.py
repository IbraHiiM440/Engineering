from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please use CSV or Excel.'}), 400
        
        # Basic info about the dataset
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'head': df.head().to_dict('records')
        }
        
        # Store data in session (in production, use proper storage)
        global current_data
        current_data = df
        
        return jsonify({'success': True, 'data_info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_sample_data')
def generate_sample_data():
    try:
        # Generate sample classification dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Features
        age = np.random.normal(35, 10, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        score = np.random.normal(650, 100, n_samples)
        
        # Target (loan approval based on features)
        target = ((age > 25) & (income > 40000) & (score > 600)).astype(int)
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'credit_score': score,
            'loan_approved': target
        })
        
        global current_data
        current_data = df
        
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'head': df.head().to_dict('records')
        }
        
        return jsonify({'success': True, 'data_info': info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize_data', methods=['POST'])
def visualize_data():
    try:
        global current_data
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        chart_type = request.json.get('chart_type')
        column = request.json.get('column')
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'histogram':
            plt.hist(current_data[column].dropna(), bins=30, alpha=0.7, color='steelblue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        
        elif chart_type == 'boxplot':
            plt.boxplot(current_data[column].dropna())
            plt.title(f'Box Plot of {column}')
            plt.ylabel(column)
        
        elif chart_type == 'correlation':
            numeric_cols = current_data.select_dtypes(include=[np.number]).columns
            corr_matrix = current_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        global current_data
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        model_type = request.json.get('model_type')
        target_column = request.json.get('target_column')
        feature_columns = request.json.get('feature_columns')
        
        # Prepare data
        X = current_data[feature_columns]
        y = current_data[target_column]
        
        # Handle categorical variables
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = le.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest_classifier':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            result = {'accuracy': accuracy, 'type': 'classification'}
            
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            result = {'accuracy': accuracy, 'type': 'classification'}
            
        elif model_type == 'svm_classifier':
            model = SVC(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            result = {'accuracy': accuracy, 'type': 'classification'}
            
        elif model_type == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            result = {'mse': mse, 'rmse': np.sqrt(mse), 'type': 'regression'}
            
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            result = {'mse': mse, 'rmse': np.sqrt(mse), 'type': 'regression'}
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_columns, model.feature_importances_))
            result['feature_importance'] = importance
        
        return jsonify({'success': True, 'results': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cluster_data', methods=['POST'])
def cluster_data():
    try:
        global current_data
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        n_clusters = request.json.get('n_clusters', 3)
        feature_columns = request.json.get('feature_columns')
        
        # Prepare data
        X = current_data[feature_columns]
        
        # Handle categorical variables
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        if len(feature_columns) >= 2:
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                       c='red', marker='x', s=200, alpha=0.8, label='Centroids')
            plt.xlabel(f'{feature_columns[0]} (scaled)')
            plt.ylabel(f'{feature_columns[1]} (scaled)')
            plt.title(f'K-Means Clustering (k={n_clusters})')
            plt.legend()
            plt.colorbar()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_stats[f'Cluster {i}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.mean(cluster_mask) * 100)
            }
        
        return jsonify({
            'success': True, 
            'plot': plot_url,
            'cluster_stats': cluster_stats,
            'inertia': float(kmeans.inertia_)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    current_data = None
    app.run(debug=True, host='0.0.0.0', port=5000)