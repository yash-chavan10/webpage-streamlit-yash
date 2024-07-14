import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.5.5.64"])
import streamlit as st
import cv2
from fer import FER
import numpy as np
import pytz
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import warnings

warnings.filterwarnings("ignore", message="file_cache is unavailable when using oauth2client >= 4.0.0")

# Set up Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'C:/Users/vinar/Downloads/facial-detection-428818-84a103b95a6d.json'  # Replace with your JSON file path
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)

# The IDs of your spreadsheets
EMPLOYEE_SPREADSHEET_ID = '1vGf_v2uXVyEqxeYCKpkhJSplwpWfvT62mtSO1cZCEeM'
RESULTS_SPREADSHEET_ID = '1KVh-f29QrGWAZ2CwGwGs83GfFY3EnAO5608RDatTrP4'

detector = FER(mtcnn=True)

def check_employee_id(employee_id):
    result = service.spreadsheets().values().get(
        spreadsheetId=EMPLOYEE_SPREADSHEET_ID,
        range='Sheet1!A:A'
    ).execute()
    values = result.get('values', [])
    return any(employee_id in row for row in values)

def write_to_sheet(employee_id, timestamp, emotion):
    values = [[employee_id, timestamp, emotion]]
    body = {'values': values}
    result = service.spreadsheets().values().append(
        spreadsheetId=RESULTS_SPREADSHEET_ID,
        range='Sheet1!A:C',
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()

def main():
    st.title("Facial Expression Detection")

    # Initialize session state
    if 'state' not in st.session_state:
        st.session_state.state = 'id_input'
        st.session_state.employee_id = ''
        st.session_state.processed_image = None
        st.session_state.detected_emotion = ''

    # State machine
    if st.session_state.state == 'id_input':
        employee_id = st.text_input("Enter Employee ID:", value=st.session_state.employee_id)
        if st.button("Submit"):
            if check_employee_id(employee_id):
                st.session_state.employee_id = employee_id
                st.session_state.state = 'photo_capture'
                st.experimental_rerun()
            else:
                st.error("Employee ID not found. Please enter a valid Employee ID.")

    elif st.session_state.state == 'photo_capture':
        st.write(f"Employee ID: {st.session_state.employee_id}")
        picture = st.camera_input("Take a picture")
        
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            result = detector.detect_emotions(cv2_img)

            if result:
                bounding_box = result[0]["box"]
                emotions = result[0]["emotions"]

                cv2.rectangle(cv2_img,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0, 155, 255),
                              2)

                emotion = max(emotions, key=emotions.get)

                cv2.putText(cv2_img,
                            emotion,
                            (bounding_box[0], bounding_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 155, 255),
                            2)

                st.session_state.processed_image = cv2_img
                st.session_state.detected_emotion = emotion

                # Write to Google Sheet with Indian timestamp
                indian_timezone = pytz.timezone('Asia/Kolkata')
                timestamp = datetime.now(indian_timezone).strftime("%Y-%m-%d %H:%M:%S")
                write_to_sheet(st.session_state.employee_id, timestamp, emotion)

                st.session_state.state = 'result_display'
                st.experimental_rerun()
            else:
                st.error("No face detected in the image. Please try again.")

    elif st.session_state.state == 'result_display':
        st.write(f"Employee ID: {st.session_state.employee_id}")
        st.image(st.session_state.processed_image, channels="BGR")
        st.write(f"Detected emotion: {st.session_state.detected_emotion}")

        if st.button("Finish"):
            st.session_state.state = 'id_input'
            st.session_state.employee_id = ''
            st.session_state.processed_image = None
            st.session_state.detected_emotion = ''
            st.experimental_rerun()

if __name__ == "__main__":
    main()
