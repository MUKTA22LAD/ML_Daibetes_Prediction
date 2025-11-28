

---

# 🩺 Diabetes Prediction using Machine Learning

A Machine Learning project that predicts whether a person is likely to have **Diabetes** based on medical diagnostic parameters. This project includes complete data preprocessing, model building, evaluation, and prediction using a trained ML model.

---

## 🔍 Overview

The goal of this project is to build a **classification model** that predicts diabetes using patient health data such as glucose level, BMI, pregnancies, insulin level, and more. The system helps identify high-risk individuals who may need further medical diagnosis.

This project includes:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Accuracy Comparison
* Final Model Saving using Pickle
* User Input Prediction

---

## 📁 Dataset

The dataset used is the **PIMA Indians Diabetes Dataset**, containing 768 samples and 8 features.
Target variable:

* **0** → No Diabetes
* **1** → Diabetes

Dataset columns:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

---

## 🧠 Machine Learning Models Tested

The following ML algorithms were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

After comparison, the **Random Forest Classifier** gave the best accuracy.

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn
* Pickle
* Jupyter Notebook

---

## 📊 Project Workflow

### **1️⃣ Import Libraries**

Load all dependencies required for EDA, ML models, and saving the model.

### **2️⃣ Load Dataset**

Read the CSV file, check missing values, and remove anomalies.

### **3️⃣ Data Preprocessing**

* Handling zero values in Glucose, Blood Pressure, etc.
* Feature scaling using StandardScaler
* Train-test split (80/20)

### **4️⃣ Model Training**

Train multiple models and compare their accuracy.

### **5️⃣ Model Evaluation**

Metrics used:

* Accuracy
* Confusion Matrix
* Classification Report

### **6️⃣ Saving the Model**

The best model is saved as:

```
model.pkl
```

### **7️⃣ User Input Prediction**

A function is created to take user input and generate predictions.

---

## 🚀 How to Run the Project

### **➡️ 1. Install dependencies**

```
pip install -r requirements.txt
```

### **➡️ 2. Run Jupyter Notebook**

```
jupyter notebook
```

### **➡️ 3. Or run the Python script**

```
python diabetes_prediction.py
```

---

## 📁 Project Structure

```
├── data/
│   └── diabetes.csv
├── Diabetes_Prediction.ipynb
├── diabetes_prediction.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 🎯 Results

* Random Forest achieved the **highest accuracy** among all models.
* The model is stable and performs well with unseen data.

---


## 🤝 Contributing

Pull requests are welcome! If you’d like to improve the model or add features, feel free to contribute.

---

## 📧 Contact

**Author:** Mukta Lad
For suggestions or improvements, feel free to reach out.

