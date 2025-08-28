# 🩺 Diabetes Prediction App  

This is a **Machine Learning-based Diabetes Prediction App** built using **Python, Scikit-learn, and Streamlit**.  
The app uses a **Random Forest Classifier** trained on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) to predict whether a person is likely to have diabetes based on health metrics.  

---

## 🚀 Features
- Train a **Random Forest Classifier** on diabetes dataset  
- Save and load the model with **joblib**  
- **GUI with Tkinter** for desktop testing  
- **Web App with Streamlit** for interactive prediction  
- Input features include:
  - Glucose level  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin level  
  - BMI (Body Mass Index)  
  - Diabetes Pedigree Function (DPF)  
  - Age  

---

## 📊 Model Training
The dataset is split into **training and testing sets**.  
We use a **Random Forest Classifier**:  
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
