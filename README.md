# Storytelling

Introducing **Storytelling**, a dynamic application crafted to create detailed physiotherapy case studies. This app taps into the immense potential of OpenAI's GPT-3 to produce engaging stories. Yet, that's just the tip of the iceberg. We've also integrated the Eleven Labs API, transforming your stories into captivating audio narrations. And to further enhance the experience, the Replicate API is employed to generate vivid images accompanying your story. Dive into a multi-sensory storytelling journey with **Storytelling**.

## Getting Started

Jump right in with these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/avivlyweb/storytelling-.git
    ```

2. Move into the directory:
    ```bash
    cd storytelling-
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Prerequisites

Before getting started with **Storytelling**, ensure you have:

- API keys for OpenAI, Eleven Labs, and Replicate.
- These keys should be placed inside a `.env` file in the root directory of the project. Remember, always keep your `.env` confidential and never commit or share it.

## Usage

Launch **Storytelling** with:

```bash
streamlit run chat.py
