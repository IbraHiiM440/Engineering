# ML Studio - Interactive Machine Learning Platform

A beautiful, interactive web application for solving machine learning problems with an intuitive interface.

## Features

🧠 **Machine Learning Models**
- Random Forest (Classification & Regression)
- Logistic Regression
- Support Vector Machines (SVM)
- Linear Regression
- Feature importance visualization

📊 **Data Visualization**
- Interactive histograms
- Box plots
- Correlation matrices
- Real-time chart generation

🔍 **Clustering Analysis**
- K-Means clustering
- Cluster visualization
- Statistical analysis of clusters

📂 **Data Management**
- Upload CSV and Excel files
- Generate sample datasets
- Drag-and-drop file upload
- Data preview and statistics

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Open Your Browser**
   Navigate to `http://localhost:5000`

## How to Use

### 1. Data Tab
- **Upload Data**: Click the upload area or drag and drop your CSV/Excel file
- **Generate Sample Data**: Click the button to create a sample loan approval dataset
- View dataset statistics, columns, and sample data

### 2. Visualize Tab
- Select chart type (histogram, box plot, or correlation matrix)
- Choose columns to visualize
- Generate interactive charts

### 3. Model Tab
- Choose from 5 different ML algorithms
- Select target column (what you want to predict)
- Select feature columns (input variables)
- Train models and view performance metrics
- See feature importance for tree-based models

### 4. Cluster Tab
- Set number of clusters (2-10)
- Select feature columns for clustering
- View cluster visualization and statistics

## Supported File Formats

- CSV (`.csv`)
- Excel (`.xlsx`)

## Model Types

**Classification Models** (for predicting categories):
- Random Forest Classifier
- Logistic Regression
- SVM Classifier

**Regression Models** (for predicting numbers):
- Linear Regression
- Random Forest Regressor

## Example Workflow

1. **Upload your data** or generate sample data
2. **Explore** your data with visualizations
3. **Train** a machine learning model
4. **Analyze** results and feature importance
5. **Perform** clustering analysis if needed

## Sample Data

The built-in sample data includes:
- Age, Income, Credit Score (features)
- Loan Approved (target for classification)
- 1000 synthetic data points

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **File Support**: openpyxl for Excel files

## Browser Compatibility

Works on all modern browsers:
- Chrome
- Firefox  
- Safari
- Edge

## Tips

- Use **Ctrl/Cmd + Click** to select multiple columns
- **Drag and drop** files for quick upload
- **Correlation matrices** work best with numeric data
- **Feature importance** shows which variables matter most
- **Clustering** helps find hidden patterns in your data

## Troubleshooting

- **File won't upload**: Check file format (CSV/Excel only)
- **Model training fails**: Ensure you have numeric data and selected appropriate columns
- **Charts not showing**: Make sure you selected the right column type for the chart

Enjoy exploring your data with ML Studio! 🚀