import streamlit as st
import mysql.connector
import bcrypt
import logging
from mysql.connector import Error as DBError

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Adjust level as needed

# --- DATABASE CONNECTION ---
def create_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Replace with your MySQL password
            database="users_auth"
        )
    except DBError as e:
        logging.error(f"Error connecting to database: {e}")
        raise

# --- USER AUTHENTICATION FUNCTIONS ---
def get_user(username):
    conn = create_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        logging.debug(f"Retrieved user: {user}")
        return user
    except DBError as e:
        logging.error(f"Error fetching user: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def register_user(name, username, password):
    conn = create_connection()
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("INSERT INTO users (name, username, password) VALUES (%s, %s, %s)", (name, username, hashed_password))
        conn.commit()
        logging.debug(f"Registered user: {username}")
    except DBError as e:
        logging.error(f"Error registering user: {e}")
        raise
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def check_password(hashed_password, user_password):
    logging.debug(f"Hashed password from DB: {hashed_password}")
    logging.debug(f"User password input: {user_password}")
    match = bcrypt.checkpw(user_password.encode('utf-8'), hashed_password.encode('utf-8'))
    logging.debug(f"Password match: {match}")
    return match


# --- USER REGISTRATION ---
def registration_page():
    st.subheader("Create a New Account")
    name = st.text_input("Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    password_confirmation = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if password == password_confirmation:
            if get_user(username) is None:
                register_user(name, username, password)
                st.success("You have successfully created an account. You can now log in.")
            else:
                st.error("Username already exists. Please choose a different one.")
        else:
            st.error("Passwords do not match.")

# --- USER LOGIN ---
def login_page():
    st.subheader("Login to Your Apple Detection Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = get_user(username)
        if user:
            hashed_password = user['password']
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                logging.debug(f"User found: {user}")
                st.session_state['name'] = user['name']
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.experimental_rerun()  # Re-run to update the UI and move to home
            else:
                st.error("Invalid password.")
        else:
            logging.error("User not found.")
            st.error("Invalid username or password.")


# --- STREAMLIT AUTHENTICATION ---
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False

if not st.session_state['authentication_status']:
    st.sidebar.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 40px;">
            <h1>🍎</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.title("Authentication Account")
    auth_option = st.sidebar.radio("Choose Option", ["Login", "Register"])
    
    if auth_option == "Login":
        login_page()
    else:
        registration_page()
else:
    st.sidebar.title(f"Welcome {st.session_state['name']}")
    if st.sidebar.button("Logout"):
        st.session_state['authentication_status'] = False
        st.experimental_rerun()

    # Import the main application logic
    import home
    home.main()
