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
        template=f""" 
        You are an AI assistant with expertise in physiotherapy. Generate a case study related to {text}. The case study should include the following sections:

        1. Patient Profile: Provide a brief introduction to the patient, including age, gender, profession, medical history, family/environmental situation, hobbies, admission diagnosis, and medical development within the hospital or at home.

        2. Assessment: Describe the process of assessing the patient's condition, including any tests or measurements performed using clinimetric tools. Provide synthetic and numeric data related to the case.

        3. Diagnosis: Based on the assessment, provide a diagnosis for the patient's condition. Include a differential diagnosis and explanations for why other diagnoses were considered and ruled out.

        4. Treatment Plan: Outline a comprehensive treatment plan for the patient, including specific exercises, therapy modalities, lifestyle recommendations and SMART goals. The plan should consider patient's mobilization and be able to address the identified problems and achieve the desired outcomes.

        5. Expected Outcomes and Progress: Discuss the expected outcomes and goals of the treatment plan, including any potential challenges or barriers to success. Monitor the patient's progress and adjust the treatment plan as needed. Reflect on choices, question whether they were the right ones, provide yourself with feedback, and use this experience to learn and improve for future case studies.

        6. Follow-up and Monitoring: Describe how the patient's progress will be monitored and any necessary adjustments to the treatment plan. Write a discharge letter directed towards the primary care physician, considering the patient's social and environmental circumstances as well as their physical condition.

        Remember, always cite your sources and base your decisions on evidence-based practice. Provide RPS form and ITE for the case study and always seek feedback for continuous improvement.
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
        options = ["Bella", "Antoni", "Arnold", "Jesse", "Domi", "Elli", "Josh", "Rachel", "Emanuele"] 
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
