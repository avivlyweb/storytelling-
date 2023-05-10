import os

import replicate
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
    prompt = PromptTemplate(
        input_variables=["text"],
        template=""" 
        ...
        """,
        text=text  # Add this line to provide the missing "text" parameter
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(text=text)

def generate_audio(text, voice):
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio

def generate_images(story_text):
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": story_text}
    )
    return output

def app():
    st.title("Physiotherapy Case Study Generator")

    with st.form(key='my_form'):
        text = st.text_input(
            "Enter a word related to physiotherapy to generate a case study",
            max_chars=None,
            type="default",
            placeholder="Enter a word related to physiotherapy to generate a case study",
        )
        options = ["Brian", "Clara", "David", "Emily", "Julia", "Kevin", "Laura", "Maggie", "Thomas"]
        voice = st.selectbox("Select a voice", options)

        if st.form_submit_button("Submit"):
            with st.spinner('Generating case study...'):
                case_study_text = generate_story(text)
                audio = generate_audio(case_study_text, voice)

            st.audio(audio, format='audio/mp3')
            images = generate_images(case_study_text)
            for item in images:
                st.image(item)

    if not text or not voice:
        st.info("Please enter a word related to physiotherapy and select a voice")

if __name__ == '__main__':
    app()
