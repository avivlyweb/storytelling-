import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

# Set up Pubmed API endpoints and query parameters
pubmed_search_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
pubmed_fetch_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
params = {
    "db": "pubmed",
    "retmode": "json",
    "retmax": 5,
    "api_key": "your_pubmed_api_key"
}

def search_pubmed(query):
    params["term"] = f"{query} AND (systematic[sb] OR meta-analysis[pt])"
    response = requests.get(pubmed_search_endpoint, params=params)
    data = response.json()
    article_ids = data["esearchresult"]["idlist"]
    return article_ids

def fetch_pubmed(article_ids):
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(article_ids)
    }
    response = requests.get(pubmed_fetch_endpoint, params=params)
    soup = BeautifulSoup(response.text, 'xml')
    articles_data = soup.find_all("PubmedArticle")
    return articles_data

def generate_story(text, articles_info):
    prompt = PromptTemplate(
        input_variables=["text", "articles_info"],
        template="""
            You are an expert AI Physiotherapist named Charlie with a 250 years career experience. Write a comprehensive physiotherapy case study assessment and treatment plan based on the HOAC model using real ebp data based on Pubmed for {text}.
            
            Step 1: Brief Introduction of the Patient Scenario
            create and write a  personal information about the patient, including age, gender, and medical history.
            
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

            Relevant literature: {articles_info}
        """
    )
    story = LLMChain(llm=llm, prompt=prompt)
    return story.run(text=text, articles_info=articles_info)

def generate_audio(text, voice):
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio

def app():
    st.title("ESPCharlie the story teller")

    text = ""
    voice = ""

    with st.form(key='my_form'):
        text = st.text_input(
            "Enter a word to generate a story",
            max_chars=None,
            type="default",
            placeholder="Enter a case study subject to generate a Physiotherapy case study",
        )
        options = ["Bella", "Antoni", "Arnold", "Jesse", "Domi", "Elli", "Josh", "Rachel", "Sam"]
        voice = st.selectbox("Select a voice", options)

        if st.form_submit_button("Submit") and text and voice:
            with st.spinner('Generating story...'):
                # Get related articles from PubMed
                article_ids = search_pubmed(text)
                articles = fetch_pubmed(article_ids)
                articles_info = ", ".join([article.find("ArticleTitle").text for article in articles])
                story_text = generate_story(text, articles_info)
                audio = generate_audio(story_text, voice)

            st.audio(audio, format='audio/mp3')

    if not text or not voice:
        st.info("Please enter a word and select a voice")

if __name__ == '__main__':
    app()
