import os
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from elevenlabs import generate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)


def generate_story(age, occupation, diagnosis, gender):
    """Generate a physiotherapy case study using the langchain library and OpenAI's GPT-3 model."""
    prompt = PromptTemplate(
        input_variables=["age", "occupation", "diagnosis", "gender"],
        template="""
        You are an expert AI Physiotherapist named Charlie with a 250 years career experience. Write a comprehensive assessment and treatment plan based on the HOAC model for a patient with the following details:
        Age: {age}
        Occupation: {occupation}
        Diagnosis: {diagnosis}
        Gender: {gender}
        
        The process should cover full physiotherapy intake, history taking, assessment, body observation, red flags, special tests, EBP, clinical reasoning, and treatment in the scope of cultural diversity. 

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
    return story.run(age=age, occupation=occupation, diagnosis=diagnosis, gender=gender)


def generate_audio(text, voice):
    """Convert the generated story to audio using the Eleven Labs API."""
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio


def app():
    st.title("ESPCharlie the story teller")

    with st.form(key='my_form'):
        age = st.text_input("Enter patient's age")
        occupation = st.text_input("Enter patient's occupation")
        diagnosis = st.text_input("Enter patient's diagnosis")
        gender = st.text_input("Enter patient's gender")
        voice = st.selectbox("Select a voice for the story", ["Joanna", "Salli", "Kendra", "Kimberly", "Ivy", "Matthew"])
        submit_button = st.form_submit_button(label='Generate Story')

    if submit_button:
        with st.spinner("Generating story..."):
            story = generate_story(age, occupation, diagnosis, gender)
            audio = generate_audio(story, voice)
        st.audio(audio, format='audio/ogg')
        st.write(story)


if __name__ == "__main__":
    app()
