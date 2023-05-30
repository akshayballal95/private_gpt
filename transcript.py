from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import  VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

video_links = ["9lVj_DZm36c", "ZUN3AFNiEgc", "8KtDLu4a-EM"]

openai_api_key = os.environ.get('OPENAI_API_KEY')


if os.path.exists('transcripts'):
    print('Directory already exists')
else:
    os.mkdir('transcripts')
for video_id in video_links:
    dir = os.path.join('transcripts', video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    df = pd.DataFrame.from_records(transcript)
    df['text']
    with open(dir+'.txt', 'w') as f:
        for line in df['text']:
            f.write(f"{line}\n")

loader = DirectoryLoader(path='./', glob = "**/*.txt", loader_cls=TextLoader,
                         show_progress=True)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

while True:
    print()
    question = input("Question: ")
    result = index.query(question)
    print()
    print(result)