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
├── data/                            # 📊 Datasets used for model training & evaluation
│   ├── smf_30.csv                   # ⛔ Realistic – Bad performance data
│   ├── smf_70.csv                   # ✅ Realistic – Good performance data
│   ├── smf_72.csv                   # ✅ Realistic – Good performance data
│   ├── Realistic_Mainframe_Performance_Dataset.csv  # ✅ Full realistic dataset
│   ├── test_indicates.txt           # 📋 Test indicators or labels
│   └── simulated_data.csv           # 🧪 Simulated – Mixed performance
│
├── Matlab/                          # 📈 MATLAB analysis and visualizations
│   ├── Comparision.m                # 📊 Performance comparison plot
│   ├── CPU_Real.m                   # 🧠 Real-world CPU performance analysis
│   ├── CPU_Simulated.m              # 🧪 Simulated CPU analysis
│   ├── Memory_Real.m                # 💾 Real-world memory analysis
│   └── Memory_Simulated.m           # 💾 Simulated memory analysis
│
├── static/                          # 🎨 Static assets for web UI
│   └── css/
│       └── style.css                # 🌈 Custom CSS styles
│
├── templates/                       # 🖼 HTML templates used by Flask
│   └── index.html                   # 🌍 Main user interface
│
├── main.py                          # 🚀 Main Python app (Flask + ML logic)
├── requirements.txt                 # 📦 Python dependencies list
└── README.md                        # 📘 Project documentation
```

