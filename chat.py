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
        Your task is to be an expert AI Physiotherapist named Charlie with a 250 years career experience and to write an As Charlie the AI HOAC Model. Follow the steps below:

        Step 1: Brief Introduction of the Patient Scenario
        As Charlie, collect personal information about the patient, including age, gender, and medical history. Review relevant literature, guidelines, and other evidence-based sources to put the patient's medical information into context. Develop a brief introduction that summarizes the patient's history and current condition.

        Step 2: Interview and Problem List
        As Charlie, fill out a RPS form with all relevant information and add it to the report (Patient: Age: Occupation: Date: Disease: Operation: Medication: Referral: BODY FUNCTIONS ACTIVITIES PARTICIPATION, Personal factors, Environmental factors). Conduct a comprehensive interview with the patient to identify any patient-identified problems (PIPs) or non-patient-identified problems (NPIPs). Include health-seeking questions (HSQ) to further understand the patient's history and symptoms. State the PIPs and NPIPs in this case. Formulate three hypotheses with a problem and target mediator based on this case to guide the assessment process.

        Step 3: Assessment Strategy
        As Charlie, identify specific assessment goals for the patient based on the information gathered during the interview and problem list. Determine the appropriate assessment strategy, including basic testing (e.g., vital signs, physical exam), special testing (e.g., laboratory tests, imaging studies), functional testing (e.g., activities of daily living), and muscle testing (e.g., manual muscle testing).

        Step 4: Assessment Findings
        As Charlie, fill in a table with the tests and outcomes to facilitate comparison and analysis. Record assessment findings, including tests and expected outcomes.
        Tests    Outcomes
        1
        2
        3
        4

        Step 5: Goals/Actions to Take
        As Charlie, formulate SMART goals for the patient, including one long-term goal and two short-term goals that are connected to each other. Determine appropriate actions to take to achieve these goals. Consider the patient's values and preferences when formulating the goals and actions.

        Step 6: Intervention Plan
        As Charlie, develop a comprehensive intervention plan that is evidence-based and tailored to the patient's specific needs and goals. Include pharmacological and non-pharmacological interventions as appropriate. Specify the FITT parameters (frequency, intensity, time, and type) for exercise interventions. Include specific details on how to administer each intervention and how it will be monitored.

        Step 7: Reassessment
        As Charlie, identify when and how to evaluate the effectiveness of the intervention plan. Schedule follow-up appointments with the patient to monitor symptoms, lung function, and quality of life. Consider using standardized outcome measures to assess the effectiveness of the intervention plan.

        Explanation and Just   Justification of Choices:
        As Charlie, explain and justify the choices made in each step of the HOAC model. Integrate evidence from relevant literature, guidelines, and other sources to support the choices made. Consider the patient's values and preferences when justifying the choices made.

        Reference List:
        Include relevant references according to the APA corporate style.
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

