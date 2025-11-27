import streamlit as st
from groq import Groq
import PyPDF2
import requests

# ---------------------------------------------------------
#   PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="FlashAI", layout="centered")

st.title("⚡ FlashAI — Ultra-Fast AI Assistant")
st.caption("Powered by Groq LLaMA 3.1 — Streaming + PDF + Weather + Task Planner")

# ---------------------------------------------------------
#   LOAD GROQ CLIENT
# ---------------------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------------------------------------------------
#   MEMORY
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
#   TOOL: WEATHER
# ---------------------------------------------------------
def get_weather(city):
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        geo = requests.get(url).json()
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current_weather=true"
        )
        weather = requests.get(weather_url).json()
        w = weather["current_weather"]

        return (
            f"🌤 **Weather in {city}:** "
            f"{w['temperature']}°C, Wind {w['windspeed']} km/h"
        )
    except:
        return "❌ Could not fetch weather."


# ---------------------------------------------------------
#   TOOL: PDF PARSER
# ---------------------------------------------------------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "")
    return text.replace("\n", " ").strip()


# ---------------------------------------------------------
#   STREAMING LLAMA MODEL
# ---------------------------------------------------------
def stream_llama(prompt):
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        stream=True,
        messages=st.session_state.messages + [{"role": "user", "content": prompt}],
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            token = getattr(chunk.choices[0].delta, "content", "")
            yield token or ""


# ---------------------------------------------------------
#   SIDEBAR TOOLS
# ---------------------------------------------------------
st.sidebar.header("🛠 FlashAI Tools")

# ----------- PDF SUMMARY TOOL -----------
pdf_file = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])

if pdf_file and st.sidebar.button("Summarize PDF"):
    pdf_text = extract_pdf_text(pdf_file)

    if not pdf_text.strip():
        st.sidebar.error("⚠ The PDF has no readable text.")
    else:
        st.sidebar.success("PDF extracted!")

        st.session_state.messages.append({
            "role": "user",
            "content": "Summarize the uploaded PDF."
        })

        with st.chat_message("assistant"):
            placeholder = st.empty()
            summary = ""
            for token in stream_llama(f"Summarize this:\n{pdf_text}"):
                summary += token
                placeholder.write(summary)

        st.session_state.messages.append({"role": "assistant", "content": summary})


# ----------- WEATHER TOOL -----------
city = st.sidebar.text_input("🌦 Weather (City)")
if st.sidebar.button("Get Weather"):
    st.sidebar.success(get_weather(city))


# ----------- TASK PLANNER TOOL (NEW) -----------
st.sidebar.subheader("📅 Task Planner / Study Plan")
task_request = st.sidebar.text_area("Describe your task or plan needed:")

if st.sidebar.button("Generate Plan"):
    if not task_request.strip():
        st.sidebar.error("Enter a task or plan request.")
    else:
        st.sidebar.success("Generating your plan...")

        # Add user's planning request to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": f"Create a structured task plan for: {task_request}"
        })

        # AI streaming response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            plan = ""

            plan_prompt = f"""
You are a Task Planning AI.

Create a detailed, organized task plan for:

"{task_request}"

Include:
- Timetable
- Daily tasks
- Weekly plan (if needed)
- Deadlines
- Learning milestones
- Structured bullet points
- Step-by-step breakdown
            """

            for token in stream_llama(plan_prompt):
                plan += token
                placeholder.write(plan)

        # Save plan to history
        st.session_state.messages.append({"role": "assistant", "content": plan})


# ---------------------------------------------------------
#   DISPLAY CHAT HISTORY
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------------------------------------------------
#   MAIN CHAT INPUT
# ---------------------------------------------------------
prompt = st.chat_input("Type your message…")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        full = ""
        placeholder = st.empty()

        for token in stream_llama(prompt):
            full += token
            placeholder.write(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
