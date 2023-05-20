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
        Hi Aviv, let's get started with our new case study for today.

        Task 1: Comprehensive Patient History
        AI, you should gather a comprehensive patient history for {text}, including demographic data, referral diagnosis, and background information on the patient's condition. This information should be synthetic, ensuring no real patient data is used.

        Task 2: Structured Interview Protocol
        Conduct a structured interview that covers all relevant areas of {text}'s health, including physical, social, and psychological aspects. 

        Task 3: Problem List and Hypotheses
        Based on the interview, create a problem list for {text}. Generate appropriate hypotheses based on the reported symptoms and physical examination findings.

        Task 4: Assessment Tools and Measurements
        Identify appropriate assessment tools, including standardized tests and measures, to confirm or rule out the generated hypotheses. Provide numeric and synthetic data based on real evidence-based practice.

        Task 5: Setting SMART Goals
        Set short-term and long-term SMART goals for {text} with clear testing criteria to monitor progress and determine whether the goals have been achieved.

        Task 6: Intervention Strategy
        Develop an intervention strategy that includes both overall tactics and specific techniques related to different hypotheses for {text}.

        Task 7: Reassessment and Follow-up
        Incorporate reassessment and follow-up into {text}'s treatment plan to track progress and modify the intervention strategy as necessary.

        Task 8: Reflection
        Reflect on this case, identifying areas for improvement and updating the decision-making process accordingly.

        All of these tasks are to be carried out while considering EBP, clinical reasoning, differential diagnosis, cultural diversity, internal and external factors, and psychosomatic factors. Please ensure patient data is anonymized and include at least 2 relevant references in the report.

        Lastly, create and provide the RPS form and ITE for this case study. Remember, never use the same Patient Profile twice! 
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
