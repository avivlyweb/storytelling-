import os
import requests
from bs4 import BeautifulSoup
import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

pubmed_search_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
pubmed_fetch_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
params = {
    "db": "pubmed",
    "retmode": "json",
    "retmax": 20,
    "api_key": "<Your PubMed API Key>"
}

def search_pubmed(query):
    params["term"] = query
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
    input_variables = { "text": text, "articles_info": articles_info }
    prompt = PromptTemplate(
        input_variables=input_variables,
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

        Relevant literature: {articles_info}
        """
    )
    story_text = llm.generate(prompt)
    return story_text

def app():
    st.title("AI Storytelling App")
    text = st.text_area("Enter the medical case for the physiotherapy scenario")
    if text:
        with st.spinner('Searching for relevant literature...'):
            article_ids = search_pubmed(text)
            articles = fetch_pubmed(article_ids)
            articles_info = "\n".join([str(article) for article in articles])
        with st.spinner('Generating story...'):
            story_text = generate_story(text, articles_info)
        st.write(story_text)

if __name__ == "__main__":
    app()
