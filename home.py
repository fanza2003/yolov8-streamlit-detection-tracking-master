import sqlite3
import hashlib
import PIL
import streamlit as st
from pathlib import Path  # Add this import

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local Modules
import settings
import helper

# --- USER AUTHENTICATION ---

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to verify user credentials
def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[1] == hash_password(password):
        return user[0]
    return None

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
    st.session_state['username'] = None
    st.session_state['name'] = None

if st.session_state['authentication_status'] != True:
    st.header("Login🍎")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        name = verify_user(username, password)
        if name:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.session_state['name'] = name
            st.success("Logged in successfully")
            st.experimental_rerun()
        else:
            st.error("Username/password is incorrect")
else:
    def main():
        # Initialize dark mode session state if not already set
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        # Main page heading
        st.title("Apple Detection")

        # Sidebar
        if st.sidebar.button("Logout"):
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.experimental_rerun()

        st.sidebar.title(f"Welcome {st.session_state['name']}")
        st.sidebar.header("🍎INDONESIAN APPLE")

        # Menu Options using radio buttons with icons
        menu_options = {
            "Home": "🏠 Home",
            "Detection": "🔍 Detection",
            "History": "📝 History"
        }
        selected_menu = st.sidebar.radio("Select Menu", list(menu_options.keys()), format_func=lambda x: menu_options[x])

        if selected_menu == "Home":
            st.header("Welcome to the Apple Detection and Tracking Application!")
            st.write("""
                This application will help apple farmers with detection and segmentation using the YOLOv8 model.
                
                Use the sidebar to navigate through different functionalities:
                - *Home*: Overview of the application.
                - *Detection*: Choose a model and source to start detecting or segmenting objects.
                - *History*: View the history of detections.
            """)
            col1, col2 = st.columns(2)

            with col1:
                st.image("images/apple.jpg", caption="Overview Image", use_column_width=True)

            with col2:
                st.image("images/apple detection.jpg", caption="Overview webcam", use_column_width=True)

        elif selected_menu == "Detection":
            st.sidebar.header("ML Model Config")

            # Model Options
            model_type = st.sidebar.radio("Select Task", ['Detection'])

            confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

            # Selecting Detection Or Segmentation
            if model_type == 'Detection':
                model_path = Path(settings.DETECTION_MODEL)

            # Load Pre-trained ML Model
            try:
                model = helper.load_model(model_path)
            except Exception as ex:
                st.error(f"Unable to load model. Check the specified path: {model_path}")
                st.error(ex)

            st.sidebar.header("Image/Video Config")
            source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

            source_img = None
            # If image is selected
            if source_radio == settings.IMAGE:
                source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

                col1, col2 = st.columns(2)

                with col1:
                    try:
                        if source_img is None:
                            default_image_path = str(settings.DEFAULT_IMAGE)
                            default_image = PIL.Image.open(default_image_path)
                            st.image(default_image_path, caption="Default Image", use_column_width=True)
                        else:
                            uploaded_image = PIL.Image.open(source_img)
                            st.image(source_img, caption="Uploaded Image", use_column_width=True)
                    except Exception as ex:
                        st.error("Error occurred while opening the image.")
                        st.error(ex)

                with col2:
                    if source_img is None:
                        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                        default_detected_image = PIL.Image.open(default_detected_image_path)
                        st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
                    else:
                        if st.sidebar.button('Detect Objects'):
                            res = model.predict(uploaded_image, conf=confidence)
                            boxes = res[0].boxes
                            res_plotted = res[0].plot()[:, :, ::-1]
                            st.image(res_plotted, caption='Detected Image', use_column_width=True)
                            try:
                                with st.expander("Detection Results"):
                                    for box in boxes:
                                        st.write(box.data)

                                # Save to history
                                if 'history' not in st.session_state:
                                    st.session_state.history = []
                                st.session_state.history.append({
                                    "image": source_img,
                                    "result": res_plotted,
                                    "boxes": boxes
                                })

                            except Exception as ex:
                                st.write("No image is uploaded yet!")

            elif source_radio == settings.VIDEO:
                try:
                    helper.play_stored_video(confidence, model)
                except PermissionError as e:
                    st.error(f"Permission denied: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            elif source_radio == settings.WEBCAM:
                helper.play_webcam(confidence, model)

            elif source_radio == settings.RTSP:
                helper.play_rtsp_stream(confidence, model)

            elif source_radio == settings.YOUTUBE:
                helper.play_youtube_video(confidence, model)

            else:
                st.error("Please select a valid source type!")

        elif selected_menu == "History":
            st.header("Detection History")

            if 'history' in st.session_state and st.session_state.history:
                for idx, record in enumerate(st.session_state.history):
                    st.subheader(f"Detection {idx + 1}")
                    st.image(record["image"], caption=f"Uploaded Image {idx + 1}", use_column_width=True)
                    st.image(record["result"], caption=f"Detected Image {idx + 1}", use_column_width=True)
                    with st.expander(f"Detection Results {idx + 1}"):
                        for box in record["boxes"]:
                            st.write(box.data)
            else:
                st.write("No detection history available.")

        # Add a toggle switch for dark mode in the sidebar at the bottom
        st.sidebar.markdown("---")
        st.session_state.dark_mode = st.sidebar.checkbox('Dark Mode', value=st.session_state.dark_mode)

        # Add the text below the Dark Mode checkbox
        if st.session_state.dark_mode:
            st.sidebar.markdown("""
            <p style="color: white; font-size: 12px;">❗use when streamlit in darkmode❗</p>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            [data-testid="stSidebar"] {
                background-color: #333333;
                color: #FFFFFF;
            }
            [data-testid="stHeader"] {
                background-color: #333333;
            }
            [data-testid="stMarkdownContainer"] {
                color: #FFFFFF;
            }
            [data-testid="stImage"] {
                background-color: #1E1E1E;
            }
            [data-testid="stExpander"] {
                background-color: #2E2E2E;
            }
            [data-testid="stSidebar"] img {
                background-color: #333333;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <p style="color: black; font-size: 12px;">❗use when streamlit in darkmode❗</p>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {
                background-color: #ffffe0;
                color: #000000;
            }
            [data-testid="stSidebar"] {
                background-color: #FFA62F;
                color: #000000.
            }
            [data-testid="stSidebar"] img {
                background-color: #FFA62F.
            }
            </style>
            """, unsafe_allow_html=True)

        st.sidebar.image("images/poon.png", use_column_width=True)

    if __name__ == "__main__":
        main()