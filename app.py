import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------- DATA LOADING & MODEL TRAINING -------------------
df = pd.read_csv("C:/Users/musma/OneDrive/Desktop/HeartApp/Medicaldataset.csv")

# Visualize histograms
def show_histograms():
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    ax = ax.flatten()
    for i, col in enumerate(df.columns[:8]):
        ax[i].hist(df[col])
        ax[i].set_xlabel(col)
        ax[i].set_title(f"Histogram of {col}")
    plt.tight_layout()
    st.pyplot(fig)

# Preprocessing
X = df.drop("Result", axis=1)
Y = df["Result"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, oob_score=True)
model.fit(X_train, Y_train)

# ------------------- MYSQL CONNECTION -------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="usmanabid@123",
    database="heart_prediction_app"
)
cursor = conn.cursor()

# ------------------- AUTH FUNCTIONS -------------------
def signup(username, email, password):
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        return False  # Username already exists

    cursor.execute(
        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
        (username, email, password)
    )
    conn.commit()
    return True

def login(username, password):
    cursor.execute(
        "SELECT * FROM users WHERE username=%s AND password=%s",
        (username, password)
    )
    result = cursor.fetchone()
    return result is not None

# ------------------- STREAMLIT UI -------------------
st.title("Heart Condition Prediction App")

# Session management
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Home", "Best Doctors", "Login", "Signup", "View Data Insights"])

# ------------------- HOME PAGE -------------------
if menu == "Home":
    st.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
            font-weight:bold;
            text-align:center;
        }
        .sub-font {
            font-size:20px !important;
            text-align:center;
            color: #555;
        }
        .feature-box {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        </style>
    """, unsafe_allow_html=True)

    st.image("image copy.png", use_container_width=True)

    st.markdown('<div class="big-font">Heart Health Prediction App ❤️</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-font">Get early insights into your heart health and connect with trusted professionals</div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Why Use Our App?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>📊 Smart Predictions</h4>
        Uses machine learning to analyze your medical data and predict heart-related conditions.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>🩺 Trusted Doctors</h4>
        Get connected to some of the best heart specialists in your area.
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="feature-box">
        <h4>📁 Personal Records</h4>
        View your previous prediction results in one place.
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-box">
        <h4>🔐 Secure Access</h4>
        Your data stays private and secure. Only you can access it.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👇 Get Started")
    st.info("Please sign up or login from the sidebar to begin your heart prediction journey.")


# ------------------- BEST DOCTORS PAGE -------------------
elif menu == "Best Doctors":
    st.title("Top Cardiologists Of The City 🩺")
    st.markdown("Here's a list of the best-reviewed heart specialists of the city:")

    doctors = [
        {"name": "Dr. Ahmed Khan", "hospital": "Shifa International", "experience": "15+ years"},
        {"name": "Dr. Maria Fatima", "hospital": "Agha Khan Hospital", "experience": "12 years"},
        {"name": "Dr. Usman Siddiqui", "hospital": "Doctors Hospital", "experience": "10 years"},
        {"name": "Dr. Sana Tariq", "hospital": "Indus Hospital", "experience": "8 years"},
        {"name" : "Dr. Humna Abid", "hospital" : "Ziauddin Hospital", "experience" : "9 years"}
    ]

    for doc in doctors:
        st.markdown(f"""
        - **{doc['name']}**  
          🏥 *{doc['hospital']}*  
          ⏳ *Experience: {doc['experience']}*
        """)

# ------------------- SIGNUP PAGE -------------------
if menu == "Signup":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username").strip()
    new_email = st.text_input("Email").strip()
    new_password = st.text_input("Password", type="password").strip()

    if st.button("Sign Up"):
        if signup(new_user, new_email, new_password):
            st.success("Account created successfully! Please login.")
        else:
            st.error("User already exists.")

# ------------------- LOGIN PAGE -------------------
elif menu == "Login":
    st.subheader("Login to Continue")
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()

    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ------------------- DATA INSIGHTS PAGE -------------------
elif menu == "View Data Insights":
    st.subheader("Data Overview")
    st.write(df.head())
    if st.checkbox("Show Histograms"):
        show_histograms()

    st.write(f"Model Test Accuracy: {accuracy_score(Y_test, model.predict(X_test)):.2f}")
    st.write(f"OOB Score: {model.oob_score_:.2f}")

# ------------------- PREDICTION PAGE (After Login) -------------------
if st.session_state.logged_in:
    username = st.session_state.username
    st.success(f"Welcome, {username}!")

    st.subheader("Enter Your Medical Details")
    age = st.number_input("Age", min_value=0.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender == "Male" else 0
    heart_rate = st.number_input("Heart rate", min_value=0.0)
    sbp = st.number_input("Systolic Blood Pressure", min_value=0.0)
    dbp = st.number_input("Diastolic Blood Pressure", min_value=0.0)
    blood_sugar = st.number_input("Blood Sugar", min_value=0.0)
    ck_mb = st.number_input("CK-MB", min_value=0.0)
    troponin = st.number_input("Troponin", min_value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([[age, gender_val, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin]],
                            columns=["Age", "Gender", "Heart rate", "Systolic blood pressure",
                                     "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"])

        prediction = model.predict(input_df)

        # Agar prediction iterable hai to pehla element lo warna prediction ko directly use karo
        try:
            pred_result = prediction[0]
        except (TypeError, IndexError):
            pred_result = prediction

        st.session_state['last_prediction'] = pred_result  # Save to session

        st.success(f"Predicted Result: {pred_result}")

        cursor.execute("""
            INSERT INTO user_data 
            (username, age, gender, heartrate, systolic_bp, diastolic_bp, blood_sugar, ckmb, troponin, prediction_result)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (username, age, gender, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin, pred_result))
        conn.commit()
        st.success("Prediction saved to database.")


    if st.checkbox("Show My Previous Predictions"):
        cursor.execute("SELECT age, gender, prediction_result FROM user_data WHERE username=%s", (username,))
        data = cursor.fetchall()
        for row in data:
            gender_str = "Male" if row[1] in ("Male", 1) else "Female"
            st.write(f"Age: {row[0]} | Gender: {gender_str} | Result: {row[2]}")

    if st.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.success("Logged out successfully.")
        st.rerun()















# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import mysql.connector
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# # ------------------- DATA LOADING & MODEL TRAINING -------------------
# df = pd.read_csv("C:/Users/musma/OneDrive/Desktop/HeartApp/Medicaldataset.csv")

# # Data info (can be shown on button if needed)
# # df.info()
# # df.describe()

# # Visualize histograms
# def show_histograms():
#     fig, ax = plt.subplots(2, 4, figsize=(15, 8))
#     ax = ax.flatten()
#     for i, col in enumerate(df.columns[:8]):
#         ax[i].hist(df[col])
#         ax[i].set_xlabel(col)
#         ax[i].set_title(f"Histogram of {col}")
#     plt.tight_layout()
#     st.pyplot(fig)

# # Preprocessing
# X = df.drop("Result", axis=1)
# Y = df["Result"]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, oob_score=True)
# model.fit(X_train, Y_train)

# # ------------------- MYSQL CONNECTION -------------------
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="usmanabid@123",
#     database="prediction_app"
# )
# cursor = conn.cursor()

# # ------------------- AUTH FUNCTIONS -------------------
# def signup(username, password):
#     try:
#         cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
#         conn.commit()
#         return True
#     except:
#         return False

# def login(username, password):
#     cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
#     return cursor.fetchone()

# # ------------------- STREAMLIT UI -------------------
# st.title("Heart Condition Prediction App")

# # Session management
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'username' not in st.session_state:
#     st.session_state.username = ""

# menu = st.sidebar.selectbox("Menu", ["Login", "Signup", "View Data Insights"])

# if st.session_state.logged_in:
#     username = st.session_state.username
#     st.success(f"Welcome, {username}!")

#     st.subheader("Enter Your Medical Details")
#     age = st.number_input("Age", min_value=0.0)
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     gender_val = 1 if gender == "Male" else 0
#     heart_rate = st.number_input("Heart rate", min_value=0.0)
#     sbp = st.number_input("Systolic Blood Pressure", min_value=0.0)
#     dbp = st.number_input("Diastolic Blood Pressure", min_value=0.0)
#     blood_sugar = st.number_input("Blood Sugar", min_value=0.0)
#     ck_mb = st.number_input("CK-MB", min_value=0.0)
#     troponin = st.number_input("Troponin", min_value=0.0)

#     if st.button("Predict"):
#         input_df = pd.DataFrame([[age, gender_val, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin]],
#             columns=["Age", "Gender", "Heart rate", "Systolic blood pressure",
#                      "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"])
#         prediction = model.predict(input_df)[0]
#         result = "Positive" if prediction == 1 else "Negative"
#         st.success(f"Predicted Result: {result}")

#         cursor.execute("""
#             INSERT INTO user_data 
#             (username, age, gender, heartrate, systolic_bp, diastolic_bp, blood_sugar, ckmb, troponin, prediction_result)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (username, age, gender, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin, result))
#         conn.commit()
#         st.success("Prediction saved to database.")

#     if st.checkbox("Show My Previous Predictions"):
#         cursor.execute("SELECT age, gender, prediction_result FROM user_data WHERE username=%s", (username,))
#         data = cursor.fetchall()
#         for row in data:
#             gender_str = "Male" if row[1] in ("Male", 1) else "Female"
#             st.write(f"Age: {row[0]} | Gender: {gender_str} | Result: {row[2]}")

#     if st.button("Logout"):
#         for key in st.session_state.keys():
#             del st.session_state[key]
#         st.success("Logged out successfully.")
#         st.rerun()


# elif menu == "Signup":
#     st.subheader("Create New Account")
#     new_user = st.text_input("Username")
#     new_pass = st.text_input("Password", type="password")
#     if st.button("Signup"):
#         if signup(new_user, new_pass):
#             st.success("Signup successful! Please login.")
#         else:
#             st.error("User already exists or error occurred.")

# elif menu == "Login":
#     st.subheader("Login to Continue")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         user = login(username, password)
#         if user:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success(f"Welcome, {username}!")
#             st.rerun()
#         else:
#             st.error("Invalid username or password.")

# elif menu == "View Data Insights":
#     st.subheader("Data Overview")
#     st.write(df.head())
#     if st.checkbox("Show Histograms"):
#         show_histograms()

#     st.write(f"Model Test Accuracy: {accuracy_score(Y_test, model.predict(X_test)):.2f}")
#     st.write(f"OOB Score: {model.oob_score_:.2f}")




# def evaluate_model(model):
#     print("Please enter the following details for prediction:")

#     age = input("Enter your Age ")
#     gender  = input("Gender (Male / Female)")
#     heart_rate = float(input("Heart Rate: "))
#     systolic_blood_pressure  = float(input("Systolic Blood Pressure: "))
#     diastolic_blood_pressure = float(input("Diastolic Blood Pressure: "))
#     blood_sugar = float(input("Blood Sugar: "))
#     ck_mb = float(input("CK-MB: "))
#     troponin = float(input("Troponin: "))
    
#     if gender.lower() == "male":
#         gender = 1
#     else:
#         gender = 0

#     input_df = pd.DataFrame([[age, gender, heart_rate, systolic_blood_pressure,
#                                diastolic_blood_pressure, blood_sugar, ck_mb, troponin]],
#                                columns = ["Age", "Gender", "Heart rate", "Systolic blood pressure",
#                                      "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"])
    
#     prediction = model.predict(input_df)

#     print(f"Predicted Result {prediction[0]}")

# evaluate_model(model)