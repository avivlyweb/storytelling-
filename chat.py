import os
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate, StreamlitCallbackHandler
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

def generate_story(text):
    """Generate a physiotherapy case study using the LangChain and OpenAI's GPT-3 model."""
    
    template = f"""
    You are an AI physiotherapist named Charlie. Analyze and generate a physiotherapy case study related to {text}.
    ...

    Always base your decisions on evidence-based practice and seek continuous feedback.
    """
    
    st_callback = StreamlitCallbackHandler(st.container())
    story = llm.run(template, callbacks=[st_callback])
    
    return story

def generate_audio(text, voice):
    """Convert the generated story to audio using the Eleven Labs API."""
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio

def app():
    st.title("Charlie the AI Physiotherapist")

    with st.form(key='my_form'):
        text = st.text_input(
            "Enter a topic to generate a case study",
            placeholder="E.g. 'knee injury'",
        )
        voice_options = ["Bella", "Antoni", "Bas", "Jesse", "Domi", "Elli", "Josh", "Rachel", "Emanuele"] 
        voice = st.selectbox("Select a voice for narration", voice_options)

        if st.form_submit_button("Generate Case Study"):
            if text and voice:
                with st.spinner('Generating story...'):
                    story_text = generate_story(text)
                    audio = generate_audio(story_text, voice)
                st.audio(audio, format='audio/mp3')
            else:
                st.info("Please enter a topic and select a voice")

if __name__ == '__main__':
    app()
