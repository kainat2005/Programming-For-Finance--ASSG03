import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# App Configuration
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #f7f9fc, #dfe7f1);
        color: #333;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        font-size: 2.8rem;
        text-align: center;
        color: #2e6f95;
    }
    h2, h3, h4 {
        color: #2e6f95;
    }
    .dataframe {
        border: 1px solid #ccc;
        border-radius: 8px;
        overflow: hidden;
    }
    .metric-container {
        background: #eaf4f9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown("<h1>💳 Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Detect fraudulent transactions using Machine Learning</p>",
    unsafe_allow_html=True,
)

# Sidebar - Upload Datasets
st.sidebar.title("📂 Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training Data", type="csv")
test_file = st.sidebar.file_uploader("Upload Test Data", type="csv")

# Initialize session state
keys = ['train_df', 'test_df', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'y_pred', 'y_prob', 'target_column']
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None

# Tabs for Navigation
tabs = st.tabs(["🏠 Home", "🔍 Data Preview", "🧪 ML Pipeline", "📊 Visualization", "⬇️ Download", "🧠 Clustering"])

# Home Tab
with tabs[0]:
    st.markdown("### Welcome to the Fraud Detection App!")
    st.markdown(
        """
        This app helps you detect fraudulent transactions using Machine Learning. 
        Follow the steps below:
        1. Upload both training and test datasets.
        2. Preprocess the data.
        3. Train a Logistic Regression Model.
        4. Evaluate and visualize results.
        """
    )

# Load Datasets
if train_file and test_file:
    try:
        st.session_state.train_df = pd.read_csv(train_file)
        st.session_state.test_df = pd.read_csv(test_file)
        st.sidebar.success("✅ Datasets uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load datasets: {e}")

# Data Preview Tab
with tabs[1]:
    if st.session_state.train_df is not None and st.session_state.test_df is not None:
        st.markdown("### 🔍 Training Dataset Preview")
        st.dataframe(st.session_state.train_df.head(), use_container_width=True)
        st.write(f"Training Dataset Shape: {st.session_state.train_df.shape}")
        
        st.markdown("### 🔍 Test Dataset Preview")
        st.dataframe(st.session_state.test_df.head(), use_container_width=True)
        st.write(f"Test Dataset Shape: {st.session_state.test_df.shape}")

        if st.checkbox("Show Dataset Statistics"):
            st.markdown("### 📊 Training Dataset Statistics")
            st.dataframe(st.session_state.train_df.describe(), use_container_width=True)
            
            st.markdown("### 📊 Test Dataset Statistics")
            st.dataframe(st.session_state.test_df.describe(), use_container_width=True)
            
        if st.checkbox("Show Column Names"):
            st.write("Training columns:", st.session_state.train_df.columns.tolist())
            st.write("Test columns:", st.session_state.test_df.columns.tolist())
    else:
        st.info("👈 Upload both training and test datasets to get started.")

# ML Pipeline Tab
if st.session_state.train_df is not None and st.session_state.test_df is not None:
    with tabs[2]:
        st.markdown("### 🧪 Machine Learning Pipeline")
        
        with st.expander("🧹 Step 1: Preprocess Data"):
            if st.button("Clean Missing Values"):
                st.session_state.train_df.dropna(inplace=True)
                st.session_state.test_df.dropna(inplace=True)
                st.success("✅ Missing values removed from both datasets.")
                st.session_state.X_train = None  # Reset features if data changed
                st.session_state.X_test = None

        with st.expander("🧪 Step 2: Feature Engineering"):
            # Let user select target column
            st.session_state.target_column = st.selectbox(
                "Select the target column (the variable you want to predict)",
                options=st.session_state.train_df.columns,
                index=len(st.session_state.train_df.columns)-1  # Default to last column
            )
            
            if st.button("Prepare Features and Target"):
                if (st.session_state.target_column in st.session_state.train_df.columns and 
                    st.session_state.target_column in st.session_state.test_df.columns):
                    
                    st.session_state.X_train = st.session_state.train_df.drop(st.session_state.target_column, axis=1)
                    st.session_state.y_train = st.session_state.train_df[st.session_state.target_column]
                    st.session_state.X_test = st.session_state.test_df.drop(st.session_state.target_column, axis=1)
                    st.session_state.y_test = st.session_state.test_df[st.session_state.target_column]
                    
                    st.success("✅ Features and targets prepared for both datasets.")
                    st.write(f"Using '{st.session_state.target_column}' as the target variable")
                    st.write(f"Training features shape: {st.session_state.X_train.shape}")
                    st.write(f"Test features shape: {st.session_state.X_test.shape}")
                else:
                    st.error(f"❌ Target column '{st.session_state.target_column}' not found in both datasets!")

        with st.expander("🤖 Step 3: Train Logistic Regression Model"):
            if st.button("Train Model"):
                if st.session_state.X_train is not None and st.session_state.y_train is not None:
                    try:
                        model = LogisticRegression(max_iter=1000)
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        st.session_state.model = model
                        st.success("✅ Model trained successfully!")
                    except Exception as e:
                        st.error(f"❌ Error training model: {e}")
                else:
                    st.error("❌ Please prepare the features and target first.")

        with st.expander("📈 Step 4: Evaluate Model"):
            if st.session_state.model is not None:
                if st.button("Evaluate"):
                    if st.session_state.X_test is not None and st.session_state.y_test is not None:
                        try:
                            y_pred = st.session_state.model.predict(st.session_state.X_test)
                            y_prob = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
                            st.session_state.y_pred = y_pred
                            st.session_state.y_prob = y_prob

                            st.markdown("#### Confusion Matrix")
                            st.dataframe(confusion_matrix(st.session_state.y_test, y_pred))

                            st.markdown("#### Classification Report")
                            st.text(classification_report(st.session_state.y_test, y_pred))

                            st.metric("🎯 ROC AUC Score", f"{roc_auc_score(st.session_state.y_test, y_prob):.2f}")
                            st.success("✅ Evaluation complete.")
                        except Exception as e:
                            st.error(f"❌ Error during evaluation: {e}")
                    else:
                        st.error("❌ Test data is not available for evaluation!")
            else:
                st.warning("⚠️ Please train a model first.")

# Visualization Tab
with tabs[3]:
    if st.session_state.train_df is not None:
        st.markdown("### 📊 Data Visualization")
        
        # Only show numeric columns for visualization
        numeric_cols = st.session_state.train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select a column to visualize", numeric_cols)
            
            if st.session_state.target_column and st.session_state.target_column in st.session_state.train_df.columns:
                fig = px.histogram(
                    st.session_state.train_df,
                    x=selected_col,
                    color=st.session_state.target_column,
                    title=f"Distribution of {selected_col} by Target",
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(
                    st.session_state.train_df,
                    x=selected_col,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found for visualization.")

# Download Tab
with tabs[4]:
    if st.session_state.y_pred is not None:
        st.markdown("### ⬇️ Download Predictions")
        
        # Create a dataframe with predictions
        results_df = st.session_state.test_df.copy()
        results_df['Prediction'] = st.session_state.y_pred
        results_df['Prediction_Probability'] = st.session_state.y_prob
        
        st.download_button(
            label="Download Predictions as CSV",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )
    else:
        st.info("Run the model evaluation first to generate predictions for download.")

# Clustering Tab
with tabs[5]:
    if st.session_state.train_df is not None:
        st.markdown("### 🧠 Transaction Clustering")
        
        numeric_cols = st.session_state.train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis feature", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis feature", numeric_cols, index=1)
            
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            
            if st.button("Run Clustering"):
                try:
                    kmeans = KMeans(n_clusters=n_clusters)
                    clusters = kmeans.fit_predict(st.session_state.train_df[[x_axis, y_axis]])
                    
                    fig = px.scatter(
                        st.session_state.train_df,
                        x=x_axis,
                        y=y_axis,
                        color=clusters,
                        title=f"K-Means Clustering (k={n_clusters})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Clustering error: {e}")
        else:
            st.warning("Need at least 2 numeric columns for clustering")