# Mainframe Resource Prediction using Machine Learning
This project uses machine learning methods, mainly **Random Forest** and **Linear Regression**, to determine and estimate how well mainframe systems will work, with metrics like **CPU utilisation**, **memory consumption**, and **transaction throughput**.  To check how accurate and reliable a model is, both simulated and real data set are used.

##  Features
- **Simulated & Realistic Datasets** for model evaluation.
- **Modeling with Random Forest & Linear Regression**
- **Error Metrics**: MSE (Mean Squared Error), MAE (Mean Absolute Error), and R² (Coefficient of Determination)
- **Performance Summary Visualization**: Compare models across all metrics and resources.
- **MATLAB Integration**: Detailed scripts to compute metrics and visualize predictions.
- **Python Program** (`main.py`):
  - Predicts performance using Random Forest
  - Evaluates models using multiple metrics
  - Visualizes predictions vs. actual values
  - Offers comparison plots across different resources


## 📁 Project Structure

```
mainframe-performance-analysis/
│
├── data/                            
│   ├── smf_30.csv                   
│   ├── smf_70.csv                 
│   ├── smf_72.csv                  
│   ├── Realistic_Mainframe_Performance_Dataset.csv  
│   ├── test_indicates.txt           
│   └── simulated_data.csv          
│
├── static/                       
│   └── css/
│       └── style.css             
│
├── templates/                       
│   └── index.html                   
│
├── main.py                          
├── requirements.txt               
└── README.md                       
```
## 🚀 How to Run the Application
### Prerequisites

Ensure the following are installed:
- Python 3.7+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/xkyleann/MainframePerformanceAnalyzer.git
cd MainframePerformanceAnalyzer
```

### 2. Create and Activate Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install flask pandas numpy scikit-learn
```

### 4. Run the Flask App

```bash
python app.py
```

Access the app at `http://127.0.0.1:5000/`

---

## 📊 Using the App

- **Upload CSV**: Format must include columns:
  - `Timestamp`
  - `CPU_Utilization`
  - `Memory_Usage`
  - `Transaction_Throughput`

- **Manual Input**: Optionally evaluate performance by entering metrics manually.

---


## ML Training 

In the funtion:
```python
def train_models(data):
```
- Trainf two types of regression models, `LinearRegression` (from `sklearn.linear_model`) and `RandomForestRegressor` (from `sklearn.ensemble`)
- Targets are `CPU_Utilization`, `Memory_Usage` and `Transaction_Throughput`

They calculate Mean Squared Error (MSE) for each model/target combination and make predictions for both target variables using both models. When it trains, the training happens when a user uploads a CSV file via the web interface. So flow has to start with user upload a `.csv` file, Flask saves the file and calls `analyze_data()`, then inside `analyze_data()`, `train_models(data)` is called and in the end ML models are trained and evulated in the uploaded data. 


## 📌 Summary

| Feature                     | Present in code? |
|----------------------------|------------------|
| Trains ML models           | ✅ Yes           |
| Uses `scikit-learn` models | ✅ Yes           |
| Trains on uploaded data    | ✅ Yes           |
| Saves/tracks models        | ❌ No            |
| Does inference             | ✅ Yes           |
| Evaluates model performance| ✅ Yes (MSE)     |



