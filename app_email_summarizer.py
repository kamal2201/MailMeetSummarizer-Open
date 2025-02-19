import streamlit as st
import os
import pickle
import base64
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

TOKEN_DIR = "user_tokens"
os.makedirs(TOKEN_DIR, exist_ok=True)


def authenticate_gmail():
    """Authenticate user with OAuth and store individual tokens."""
    token_path = os.path.join(TOKEN_DIR, ".pickle")
    creds = None

    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as token_file:
            pickle.dump(creds, token_file)

    return build("gmail", "v1", credentials=creds)


def fetch_email_threads(service, max_results=10):
    """Fetch emails and group them by ThreadId."""
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = results.get("messages", [])

    thread_map = {}

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        thread_id = msg_data["threadId"]
        payload = msg_data["payload"]
        headers = payload["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")

        if "parts" in payload:
            parts = payload["parts"]
            email_body = next((base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                               for part in parts if part.get("body", {}).get("data")), "No Content")
        else:
            email_body = "No Content"

        if thread_id not in thread_map:
            thread_map[thread_id] = {
                "subject": subject,
                "messages": []
            }

        thread_map[thread_id]["messages"].append(f"From: {sender}\n{email_body}\n")

    return thread_map


def summarize_thread(thread):
    """Summarizes an email thread using Azure OpenAI."""
    conversation = "\n".join(thread["messages"])

    if not conversation.strip():
        return "No conversation found."

    messages = [
        SystemMessage(content="Summarize the following email thread in 3-4 sentences:"),
        HumanMessage(content=conversation)
    ]

    response = llm.invoke(messages)

    return response.content.strip()


def main():
    st.title("Gmail Thread Summarizer")

    service = authenticate_gmail()
    email_threads = fetch_email_threads(service)

    if not email_threads:
        st.write("No email threads found.")
        return

    for thread_id, thread in email_threads.items():
        summary = summarize_thread(thread)
        st.subheader(thread["subject"])
        st.write(f"**Thread ID:** {thread_id}")
        st.write(f"**Thread Summary:** {summary}")
        st.markdown("---")


if __name__ == "__main__":
    main()
