from transformers import pipeline

# Load summarizer from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_user_problem(conversation: str) -> dict:
    # Step 1: Extract only user messages
    user_lines = [line.replace("User:", "").strip() 
                  for line in conversation.splitlines() if line.startswith("User:")]
    user_text = " ".join(user_lines)

    # Step 2: Summarize user queries only
    summary = summarizer(user_text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']

    # Step 3: Title = First short sentence
    title = summary.split('.')[0].strip()

    return {
        "title": title,
        "description": summary
    }

# # 🧪 Example
# conversation = """
# User: I tried resetting my password but didn't get the email.
# Assistant: Please check your spam folder.
# User: I did, it's not there. Can you help me get it resent?
# Assistant: Sure, let me resend the email.
# """

# result = extract_user_problem(conversation)
# print(result)
