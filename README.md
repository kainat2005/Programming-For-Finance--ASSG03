# 💳 Credit Card Fraud Detection App

This is a web-based app built using **Streamlit** that enables users to detect fraudulent credit card transactions using **Logistic Regression**. It also supports basic **clustering** (KMeans), visualizations (using Plotly), and dataset analysis.

---

## 🚀 Features

- Upload training and testing datasets (CSV)
- View dataset previews and statistics
- Handle missing data
- Select target column
- Train Logistic Regression model
- Evaluate using Confusion Matrix, Classification Report, ROC AUC Score
- Visualize feature distributions
- Perform clustering
- Download predictions

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend/ML:** Scikit-learn
- **Data Handling:** Pandas
- **Visualizations:** Plotly

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/fraud-detection-app.git
cd fraud-detection-app
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
🧪 Sample Usage
Upload your training and test datasets in CSV format via the sidebar.

Clean missing values (optional).

Select the target column.

Train the logistic regression model.

Evaluate and visualize results.

Download predictions.

📁 Folder Structure
bash
Copy
Edit
fraud-detection-app/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
📌 Notes
Make sure both datasets (train/test) contain the same columns including the target column.

Recommended to preprocess datasets before uploading (e.g., scaling, encoding if needed).

📜 License
This project is open-source and available under the MIT License.
