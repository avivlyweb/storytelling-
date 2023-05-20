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
    """Generate a physiotherapy case study using the langchain library and OpenAI's GPT-3 model."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=""" 
        You are an expert AI Physiotherapist named Charlie with a 250 years career experience. Write a comprehensive assessment and treatment plan based on the HOAC model for {text}.
        
        Step 1: Brief Introduction of the Patient Scenario
        Collect personal information about the patient, including age, gender, and medical history.
        
        Step 2: Interview and Problem List
        Fill out a RPS form and conduct a comprehensive interview with the patient to identify any patient-identified problems (PIPs) or non-patient-identified problems (NPIPs).
        Formulate three hypotheses with a problem and target mediator based on this case to guide the assessment process.

        Step 3: Assessment Strategy
        Identify specific assessment goals for the patient and determine the appropriate assessment strategy, including basic testing, special testing, functional testing, and muscle testing.

        Step 4: Assessment Findings
        Record assessment findings, including tests and expected outcomes.

        Step 5: Goals/Actions to Take
        Formulate SMART goals for the patient and determine appropriate actions to take to achieve these goals.

        Step 6: Intervention Plan
        Develop a comprehensive intervention plan that includes pharmacological and non-pharmacological interventions as appropriate.

        Step 7: Reassessment
        Identify when and how to evaluate the effectiveness of the intervention plan. Schedule follow-up appointments with the patient to monitor symptoms, function, and quality of life.

        Explanation and Justification of Choices:
        Explain and justify the choices made in each step of the HOAC model, integrating evidence from relevant literature, guidelines, and other sources to support the choices made.
                 """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(text=text)


def generate_audio(text, voice):
    """Convert the generated story to audio using the Eleven Labs API."""
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio


def generate_images(story_text):
    """Generate images using the story text using the Replicate API."""
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": story_text}
    )
    return output


def app():
    st.title("ESPCharlie the story teller")

    with st.form(key='my_form'):
        text = st.text_input(
            "Enter a word to generate a story",
            max_chars=None,
            type="default",
            placeholder="Enter a case study subject to generate a Physiotherapy case study",
        )
        options = ["Bella", "Antoni", "Arnold", "Jesse", "Domi", "Elli", "Josh", "Rachel", "Sam"]
        voice = st.selectbox("Select a voice", options)

        if st.form_submit_button("Submit"):
            with st.spinner('Generating story...'):
                story_text = generate_story(text)
                audio = generate_audio(story_text, voice)

            st.audio(audio, format='audio/mp3')
            images = generate_images(story_text)
            for item in images:
                st.image(item)

    if not text or not voice:
        st.info("Please enter a word and select a voice")


if __name__ == '__main__':
    app()
