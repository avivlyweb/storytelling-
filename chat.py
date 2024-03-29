import os
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

def generate_story(text):
    """Generate a physiotherapy case study using the langchain library and OpenAI's GPT-3 model."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=""" 
        You are an AI physiotherapist named Charlie. Analyze and generate a physiotherapy case study related to {text}.

        **Patient Profile**: Briefly describe the patient's background and medical history.

        **Assessment**: Detail your initial assessment and findings.

        **Diagnosis**: Based on your assessment, provide a diagnosis and formulate hypotheses.

        **Treatment Goals**: Establish treatment goals linked to your findings.

        **Intervention Plan**: Outline an intervention plan and explain the reasons for your choices.

        **Expected Outcomes**: Describe the anticipated outcomes and how you'll monitor progress.

        **Reflection**: Reflect on your choices and consider any necessary adjustments. 

        Always base your decisions on evidence-based practice and seek continuous feedback.
        """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(text=text)

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
