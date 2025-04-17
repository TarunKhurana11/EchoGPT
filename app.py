import gradio as gr
from openai import OpenAI
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Transcribe voice input
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service error."

# Generate response from OpenAI and gTTS
def generate_response(history, text_input, audio_input):
    if audio_input:
        text_input = transcribe_audio(audio_input)

    if not text_input.strip():
        return history, "Please say or type something.", None

    # Build conversation history for OpenAI
    messages = [{"role": "system", "content": "You are a helpful and friendly AI assistant."}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": text_input})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"OpenAI API Error: {str(e)}"

    # Create audio reply
    filename = f"response_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=reply, lang='en')
    tts.save(filename)

    # Update chat history in message format
    history.append({"role": "user", "content": text_input})
    history.append({"role": "assistant", "content": reply})

    return history, "", filename

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Voice/Text Chatbot\nChat using your **voice** or **keyboard**, and hear answers spoken back to you!")

    chatbot = gr.Chatbot(label="Chat History", height=400, type="messages")

    with gr.Row():
        text_input = gr.Textbox(placeholder="Type your message here...", scale=3)
        audio_input = gr.Audio(type="filepath", label="🎙️ Record", scale=2)

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    audio_output = gr.Audio(label="🔊 Bot's Voice", autoplay=True)
    state = gr.State([])

    send_btn.click(
        fn=generate_response,
        inputs=[state, text_input, audio_input],
        outputs=[chatbot, text_input, audio_output]
    )

    clear_btn.click(
        fn=lambda: ([], "", None),
        inputs=None,
        outputs=[chatbot, text_input, audio_output, state]
    )

demo.launch(share=True)
