import os
import json
import sendgrid
from sendgrid.helpers.mail import Mail, Email, Personalization
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
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

app = Flask(__name__)

email_threads = {}


def extract_email_details(input_text):
    """
    Uses Azure OpenAI GPT-4 to extract recipient, subject, and body from user input.
    """
    try:
        system_prompt = """
        You are an AI that extracts email details from a given text. 
        Extract the recipient email, subject, and body.
        Ensure the output is strictly in JSON format.

        Example Output:
        {
            "recipient": "john@example.com",
            "subject": "Meeting Update",
            "body": "Please join the meeting at 3 PM today."
        }
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_text)
        ]

        response = llm.invoke(messages)

        extracted_data = json.loads(response.content.strip())
        return extracted_data

    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON - Response: {response.content.strip()}")
        return None

    except Exception as e:
        print(f"Error extracting email details: {e}")
        return None


def generate_email(recipient, subject, context):
    """
    Uses Azure OpenAI GPT-4 to generate a professional email based on context.
    """
    try:
        greeting = f"Dear {recipient.split('@')[0].title()}," if "@" in recipient else "Hello,"

        system_prompt = f"""
        You are an expert in drafting professional emails. Based on the provided context, 
        generate a well-structured email **excluding the subject** (since it is already specified separately).

        The email should include:
        - **Greeting:** {greeting}
        - **Body:** Clearly convey the purpose in a professional manner.
        - **Closing:** A polite closing statement, such as 'Best regards, [Your Name]'.

        **Do not include the subject line in the generated email.**
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]

        response = llm.invoke(messages)

        return response.content.strip()

    except Exception as e:
        print(f"Error generating email: {e}")
        return f"Error: Could not generate email due to {str(e)}"


def send_email_sendgrid(to_email, subject, body):
    """
    Sends an email using SendGrid API and tracks email threads.
    """
    try:
        sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)

        # Check if an email thread already exists for this recipient
        thread_id = email_threads.get(to_email)

        if not thread_id:
            # Generate a new thread ID (UUID-based)
            thread_id = f"thread-{os.urandom(8).hex()}"
            email_threads[to_email] = thread_id  # Store the new thread

        message = Mail(
            from_email=Email(SMTP_EMAIL),
            subject=subject,
            plain_text_content=body
        )

        # ✅ Correct way to add custom arguments
        personalization = Personalization()
        personalization.add_to(Email(to_email))
        personalization.add_custom_arg(sendgrid.helpers.mail.CustomArg("thread_id", thread_id))

        message.add_personalization(personalization)

        response = sg.send(message)

        return {
            "status": "success",
            "message": f"Email sent to {to_email}",
            "response_code": response.status_code,
            "thread_id": thread_id
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route('/send-email', methods=['POST'])
def api_send_email():
    """
    API endpoint to send an email based on natural language input.
    """
    try:
        data = request.json
        user_input = data.get('input_text')

        if not user_input:
            return jsonify({"status": "error", "message": "Input text is required"}), 400

        # Extract recipient, subject, and body using GPT
        email_details = extract_email_details(user_input)

        if not email_details:
            return jsonify({"status": "error", "message": "Failed to extract email details"}), 500

        recipient = email_details.get("recipient")
        subject = email_details.get("subject", "Automated Email")
        context = email_details.get("body")

        if not recipient or not context:
            return jsonify({"status": "error", "message": "Incomplete email details extracted"}), 500

        email_body = generate_email(recipient, subject, context)

        response = send_email_sendgrid(recipient, subject, email_body)
        print(response)
        return jsonify(response)

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
