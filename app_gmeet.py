import os
import json
import pickle
import streamlit as st
import openai
import google_auth_oauthlib.flow
import googleapiclient.discovery
import dateparser
from google.auth.transport.requests import Request
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

CREDENTIALS_FILE = "credentials/client_secret.json"
TOKEN_DIR = "user_tokens"

# Google Calendar API Scopes (OAuth)
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

os.makedirs(TOKEN_DIR, exist_ok=True)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    azure_deployment="Alfred-gpt-4-o-mini",
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def authenticate_google():
    """Handles OAuth authentication for Google Calendar, storing individual tokens per user."""

    token_path = os.path.join(TOKEN_DIR, f".pickle")

    creds = None

    # Load existing token
    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    # If no valid token, start OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # Refresh token if expired
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the new token
        with open(token_path, "wb") as token_file:
            pickle.dump(creds, token_file)

    return googleapiclient.discovery.build("calendar", "v3", credentials=creds)


def extract_meeting_details(prompt):
    """Extracts meeting details using Azure GPT-4."""
    system_prompt = """
    You are an AI that extracts meeting details from text.
    Return a JSON with keys:
    - title (string)
    - date (YYYY-MM-DD)
    - start_time (HH:MM)
    - end_time (HH:MM)
    - attendees (list of emails)

    If any detail is missing, return null.
    Always extract correct dates and times in UTC format.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    try:
        extracted_data = json.loads(response.content)  # Extract JSON response
        return extracted_data
    except json.JSONDecodeError:
        return {}


def parse_datetime(date_str, time_str):
    """Converts 'tomorrow 3 PM' into 'YYYY-MM-DD HH:MM' format."""
    full_datetime_str = f"{date_str} {time_str}"
    parsed_dt = dateparser.parse(full_datetime_str)

    if parsed_dt:
        return parsed_dt.strftime("%Y-%m-%dT%H:%M:%S")  # ISO format
    return None


def schedule_meeting(event_title, date, start_time, end_time, attendees):
    """Schedules a Google Meet meeting."""
    service = authenticate_google()

    start_time_iso = parse_datetime(date, start_time)
    end_time_iso = parse_datetime(date, end_time)

    if not start_time_iso or not end_time_iso:
        st.error("❌ Failed to parse date/time. Please check inputs.")
        return None

    event = {
        "summary": event_title,
        "description": "Scheduled via AI assistant.",
        "start": {"dateTime": start_time_iso, "timeZone": "IST"},
        "end": {"dateTime": end_time_iso, "timeZone": "IST"},
        "attendees": [{"email": email.strip()} for email in attendees if email.strip()],
        "conferenceData": {
            "createRequest": {
                "requestId": f"meet-{date.replace('-', '')}{start_time.replace(':', '')}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        }
    }

    try:
        created_event = service.events().insert(
            calendarId="primary",
            body=event,
            conferenceDataVersion=1
        ).execute()

        return created_event.get("hangoutLink")
    except googleapiclient.errors.HttpError as error:
        st.error(f"❌ Google Calendar API Error: {error}")
        return None


# Streamlit UI
st.set_page_config(page_title="Google Meet Scheduler", page_icon="📅", layout="centered")

st.markdown("<h1 style='text-align: center;'>📅 Google Meet Scheduler</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-powered meeting scheduling</h4>", unsafe_allow_html=True)

st.divider()

st.markdown("### ✨ Enter your request below")
user_input = st.text_input("Example: 'Schedule a meeting with John at 3 PM tomorrow'")

if user_input:
    extracted_data = extract_meeting_details(user_input)

    if not all(extracted_data.values()):
        st.warning("⚠️ Some details are missing. Please fill in the required fields.")

        with st.form("Meeting Details"):
            title = st.text_input("Meeting Title", extracted_data.get("title", ""))
            date = st.date_input("Date")
            start_time = st.time_input("Start Time")
            end_time = st.time_input("End Time")
            attendees = st.text_area("Attendees (comma-separated emails)")

            submitted = st.form_submit_button("✅ Schedule Meeting")
            if submitted:
                extracted_data.update({
                    "title": title,
                    "date": str(date),
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "attendees": [email.strip() for email in attendees.split(",")]
                })

                meet_link = schedule_meeting(
                    extracted_data["title"],
                    extracted_data["date"],
                    extracted_data["start_time"],
                    extracted_data["end_time"],
                    extracted_data["attendees"]
                )

                if meet_link:
                    st.success(f"✅ Meeting Scheduled: [Join Meeting]({meet_link})")
    else:
        meet_link = schedule_meeting(
            extracted_data["title"],
            extracted_data["date"],
            extracted_data["start_time"],
            extracted_data["end_time"],
            extracted_data["attendees"]
        )

        if meet_link:
            st.success(f"✅ Meeting Scheduled: [Join Meeting]({meet_link})")