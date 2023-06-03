from langchain import ConversationChain, PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import  VectorstoreIndexCreator
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import GPT4All
import os
from langchain.memory import VectorStoreRetrieverMemory
from langchain.callbacks import get_openai_callback

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

video_links = ["9lVj_DZm36c", "ZUN3AFNiEgc", "8KtDLu4a-EM"]

if os.path.exists('transcripts'):
    print('Directory already exists')
else:
    os.mkdir('transcripts')
for video_id in video_links:
    dir = os.path.join('transcripts', video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    with open(dir+'.txt', 'w') as f:
     for line in transcript:
            f.write(f"{line['text']}\n")


loader = DirectoryLoader(path='./', glob = "**/*.txt", loader_cls=TextLoader,
                        show_progress=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=5))
memory = VectorStoreRetrieverMemory(retriever=retriever)

llm = GPT4All(model="./ggml-mpt-7b-instruct.bin", top_p=0.15, temp=0.5, repeat_penalty=1.1)

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Do not make up answers and provide only information that you have.
Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm, 
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
)



st.set_page_config(
    page_title="YouTubeGPT",
    page_icon=":robot:"
)

st.header("YouTubeGPT")
st.markdown("[Github](https://github.com/akshayballal95)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 

user_input = get_text()

if user_input:
    with get_openai_callback() as cb:
        output=conversation_with_summary.predict(input = "Hi How are you")
        print(cb)
   

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
