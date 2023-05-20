import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from elevenlabs import generate
import streamlit as st

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

def generate_story(patient_info):
    prompt = PromptTemplate(
        input_variables={"patient_info": dict},  # Change here
        template=f""" 
        You are an expert AI Physiotherapist named Charlie with a 250 years career experience. For any case studies provided, synthetic data will only be used to support the numeric data and patient information, while the evidence-based practice (EBP) will be based on real research findings.
        You are tasked with completing a comprehensive assessment and treatment plan based on the HOAC model for {patient_info}.

        You will cover areas such as full physiotherapy intake, history taking, assessment, body observation, red flags, special tests, EBP, clinical reasoning, and even treatment planning. In the scope of cultural diversity, you will use NLU and NLP to be empathetic, take into account the personal factors of the patients, internal and external factors, and psychosomatic factors.
        
        You will also focus on providing full physiotherapy intervention, home exercises, prevention advice, and work-related problems (ergonomics). Furthermore, you will help with health care seeking questions of the patient and provide study cases and teach students clinical reasoning.

        You will use and list the clinimetrics tools, provide synthetic and numeric data when it comes to AROM, PROM, specific medications, patient's family and environmental situation, patient's hobbies, all the information that can have a significant impact on the patient's recovery. You will write in a list the numeric and synthetic data for the test results and outcome based on the case study.
        """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(patient_info=patient_info)

def generate_audio(text, voice):
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio

def app():
    st.title("ESPCharlie the story teller")

    with st.form(key='my_form'):
        age = st.text_input("Enter patient's age")
        occupation = st.text_input("Enter patient's occupation")
        diagnosis = st.text_input("Enter patient's diagnosis")
        gender = st.text_input("Enter patient's gender")
        patient_info = {"age": age, "occupation": occupation, "diagnosis": diagnosis, "gender": gender}
        
        options = ["Bella", "Antoni", "Arnold", "Jesse", "Domi", "Elli", "Josh", "Rachel", "Sam"]
        voice = st.selectbox("Select a voice", options)

        if st.form_submit_button("Submit"):
            with st.spinner('Generating story...'):
                story_text = generate_story(patient_info)
                audio = generate_audio(story_text, voice)

            st.audio(audio, format='audio/mp3')

    if not age or not occupation or not diagnosis or not gender or not voice:
        st.info("Please enter all patient details and select a voice")

if __name__ == '__main__':
    app()
