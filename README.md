# 🔥 Employee Burn Rate Prediction
 
A machine learning project to predict employee **Burn Rate** using regression models. Includes both a command-line pipeline (`main.py`) and an interactive **Streamlit** web app.
 
---
 
## 📁 Project Structure
 
```
├── main.py                  # Core ML pipeline (train, evaluate, predict)
├── app.py                   # Streamlit web application
├── train.csv                # Training dataset
├── test.csv                 # Test dataset
├── sample_submission.csv    # Submission format reference
├── submission.csv           # Generated predictions (output)
└── requirements.txt         # Python dependencies
```
 
---
 
## 📊 Dataset
 
| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train.csv` | 22,750 | 9 | Labeled employee data with Burn Rate |
| `test.csv` | 12,250 | 8 | Unlabeled employee data for prediction |
| `sample_submission.csv` | — | 2 | Expected output format |
 
### Features
 
| Column | Type | Description |
|--------|------|-------------|
| `Employee ID` | String | Unique identifier |
| `Date of Joining` | Date | Employee joining date |
| `Gender` | Categorical | Male / Female |
| `Company Type` | Categorical | Product / Service |
| `WFH Setup Available` | Categorical | Yes / No |
| `Designation` | Float | Seniority level |
| `Resource Allocation` | Float | Hours allocated |
| `Mental Fatigue Score` | Float | Self-reported fatigue (0–10) |
| `Burn Rate` ⭐ | Float | **Target variable** (0–1) |
 
---
 
## ⚙️ Installation
 
### 1. Clone the repository
 
```bash
git clone https://github.com/your-username/employee-burn-rate.git
cd employee-burn-rate
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
## 🚀 Usage
 
### Run the ML pipeline (CLI)
 
```bash
python main.py
```
 
Outputs `submission.csv` with predicted burn rates for the test set.
 
### Run the Streamlit app
 
```bash
streamlit run app.py
```
 
---
 
## 🧠 Model
 
**GradientBoostingRegressor** with the following configuration:
 
```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
```
 
### Feature Engineering
 
- Extracted `join_year`, `join_month`, `join_day` from `Date of Joining`
- Label-encoded categorical columns (`Gender`, `Company Type`, `WFH Setup Available`)
- Median imputation for missing values
---
 
## 📈 Results
 
| Metric | Score |
|--------|-------|
| Validation RMSE | **0.0714** |
| Validation R² | **0.8575** |
| 5-Fold CV RMSE | **0.0724 ± 0.0017** |
 
---
 
## 🛠️ Tech Stack
 
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
 
---
 
## 📄 License
 
This project is licensed under the [MIT License](LICENSE).
