import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, session, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import safe_join
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


CPU_THRESHOLD = 80  # above considered bad
MEMORY_THRESHOLD = 80  # above considered bad
TRANSACTION_THRESHOLD = 5000  # below considered bad for throughput


def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def train_models(data):
    """Train both Linear Regression and Random Forest Regression models."""
    try:
     
        X = data[["Transaction_Throughput"]]
        y_cpu = data["CPU_Utilization"]
        y_memory = data["Memory_Usage"]

        # Linear Regression Model
        linear_model_cpu = LinearRegression()
        linear_model_memory = LinearRegression()
        linear_model_cpu.fit(X, y_cpu)
        linear_model_memory.fit(X, y_memory)

        # Random Forest Regression Model
        rf_model_cpu = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_memory = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_cpu.fit(X, y_cpu)
        rf_model_memory.fit(X, y_memory)

        # predictions
        data["Linear_Pred_CPU"] = linear_model_cpu.predict(X)
        data["Linear_Pred_Memory"] = linear_model_memory.predict(X)
        data["RF_Pred_CPU"] = rf_model_cpu.predict(X)
        data["RF_Pred_Memory"] = rf_model_memory.predict(X)

        # MSE
        mse_linear_cpu = mean_squared_error(y_cpu, data["Linear_Pred_CPU"])
        mse_linear_memory = mean_squared_error(y_memory, data["Linear_Pred_Memory"])
        mse_rf_cpu = mean_squared_error(y_cpu, data["RF_Pred_CPU"])
        mse_rf_memory = mean_squared_error(y_memory, data["RF_Pred_Memory"])

        model_results = {
            "mse_linear_cpu": mse_linear_cpu,
            "mse_linear_memory": mse_linear_memory,
            "mse_rf_cpu": mse_rf_cpu,
            "mse_rf_memory": mse_rf_memory,
        }

        return data, model_results
    except Exception as e:
        return f"Error training models: {str(e)}", None


def analyze_data(file_path):
    """Analyze uploaded CSV data for performance issues and apply ML models."""
    try:
        data = pd.read_csv(file_path)


        data.rename(columns={"Time": "Timestamp"}, inplace=True)
        required_columns = ["Timestamp", "CPU_Utilization", "Memory_Usage", "Transaction_Throughput"]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return f"Error: Missing columns {', '.join(missing_columns)} in uploaded file!", None, None

        # train ML models
        data, model_results = train_models(data)
        if isinstance(data, str):  # Handle model training errors
            return data, None, None


        issues = []
        for row in data.itertuples(index=False):
            issue_list = []
            if row.CPU_Utilization > CPU_THRESHOLD:
                issue_list.append("High CPU usage")
            if row.Memory_Usage > MEMORY_THRESHOLD:
                issue_list.append("High memory usage")
            if row.Transaction_Throughput < TRANSACTION_THRESHOLD:
                issue_list.append("Low transaction throughput")

            if issue_list:
                issues.append({"Timestamp": row.Timestamp, "Issues": "; ".join(issue_list)})

        return None, issues, model_results
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None


def evaluate_performance(cpu, memory, throughput):
    """Evaluate system performance based on thresholds."""
    cpu_status = "Good" if 10 <= cpu <= CPU_THRESHOLD else "Bad (High CPU Usage)" if cpu > CPU_THRESHOLD else "Bad (Too Low CPU Usage)"
    memory_status = "Good" if 10 <= memory <= MEMORY_THRESHOLD else "Bad (High Memory Usage)" if memory > MEMORY_THRESHOLD else "Bad (Too Low Memory Usage)"
    throughput_status = "Good (Above Minimum Throughput Threshold)" if throughput >= TRANSACTION_THRESHOLD else f"Bad (Low Transaction Throughput - Threshold: {TRANSACTION_THRESHOLD} transactions/sec)"

    overall_status = "Bad" if any("Bad" in status for status in [cpu_status, memory_status, throughput_status]) else "Good"

    return cpu_status, memory_status, throughput_status, overall_status


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "submit_file" in request.form:
            if "file" not in request.files:
                flash("No file part", "danger")
                return redirect(url_for("index"))

            file = request.files["file"]
            if file.filename == "":
                flash("No selected file", "danger")
                return redirect(url_for("index"))

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = safe_join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                error, issues, model_results = analyze_data(file_path)
                if error:
                    flash(error, "danger")
                else:
                    session["issues"] = issues
                    session["model_results"] = model_results
                    flash("File processed successfully!", "success")

        elif "submit_evaluate" in request.form:
            try:
                cpu = float(request.form["cpu_utilization"])
                memory = float(request.form["memory_usage"])
                throughput = float(request.form["transaction_throughput"])

                cpu_status, memory_status, throughput_status, overall_status = evaluate_performance(cpu, memory, throughput)

                session.update({
                    "cpu_status": cpu_status,
                    "memory_status": memory_status,
                    "throughput_status": throughput_status,
                    "overall_status": overall_status,
                    "cpu": cpu,
                    "memory": memory,
                    "throughput": throughput,
                })

                flash("Performance evaluation completed!", "success")
            except ValueError:
                flash("Invalid input. Please enter numerical values.", "danger")

    return render_template(
        "index.html",
        issues=session.get("issues"),
        cpu_status=session.get("cpu_status"),
        memory_status=session.get("memory_status"),
        throughput_status=session.get("throughput_status"),
        overall_status=session.get("overall_status"),
        cpu=session.get("cpu"),
        memory=session.get("memory"),
        throughput=session.get("throughput"),
        model_results=session.get("model_results"),
    )


if __name__ == "__main__":
    app.run(debug=True)
