import streamlit as st 
from langchain_openai import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain

from prompts import user_prompt, system_prompt

# Constants
PAGE_TITLE = "Notes AI"
PAGE_ICON = "📝"
OPENAI_MODEL_NAME = "gpt-4"
OPENAI_API_KEY_PROMPT = 'OpenAI API Key'
PROMPT_TEMPLATE = "Write a concise summary of the following:\n{text}\nCONCISE SUMMARY:"
REFINE_TEMPLATE = "Your job is to produce a final summary\nWe have provided an existing summary up to a certain point: {existing_answer}\nWe have the opportunity to refine the existing summary" \
                  "(only if needed) with some more context below.\n------------\n{text}\n------------\n"

# Set page config
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, initial_sidebar_state="collapsed")
st.header("",divider='green')
st.title(f"📝 :green[_{PAGE_TITLE}_] | Videos Summarized")
st.header("",divider='green')

# Get OpenAI API key
openai_api_key = st.sidebar.text_input(OPENAI_API_KEY_PROMPT, type='password')

if not openai_api_key.startswith('sk-'):
    st.info("Please add your OpenAI API key in the sidebar to continue.")
    st.stop()

with st.sidebar:        
    st.divider()
    st.write('*Your notes generation flow begins with giving us the youtube video link and number of lines needed*')
    st.caption('''**That's it! 
               Once we have it, we'll understand it and start exploring our options.
                Then, we'll work together to and come up with the best possible summary of the video.**
    ''')
    st.divider()

# Form inputs
with st.form("video_info"):
    video_url = st.text_input(
            "Give us the url of the youtube video:",
            placeholder= "https://www.youtube.com/watch?v=mEsleV16qdo",
        )
    num_of_lines = st.slider('How many lines?', 2, 50, 10)
    submitted = st.form_submit_button("Generate notes!")

if submitted: 
    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False, language = ['en-US', 'en'])
        transcripts = loader.load()
    except Exception as e:
        st.error(f"Failed to load video: {str(e)}")
        st.stop()

    splitter = TokenTextSplitter(model_name=OPENAI_MODEL_NAME, chunk_size=10000, chunk_overlap=100)
    chunks = splitter.split_documents(transcripts)

    st.divider()
    st.subheader("Notes: ")
    
    with st.spinner('Please wait...'):
        llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.2, openai_api_key=openai_api_key)
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        refine_prompt = PromptTemplate.from_template(REFINE_TEMPLATE)
        
        summarize_chain = load_summarize_chain(llm=llm, chain_type="refine",question_prompt=prompt, 
                                               refine_prompt=refine_prompt, input_key="input_documents",
                                                output_key="output_text")
        
        summary = summarize_chain({"input_documents": chunks})
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        request = chat_prompt.format_prompt(
        text=summary['output_text'], 
        num=num_of_lines
        ).to_messages()
        
        result = llm(request)
        st.write(result.content)
        
        st.divider()