from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import  VectorstoreIndexCreator
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import OpenAI
import os

video_links = ["9lVj_DZm36c", "ZUN3AFNiEgc", "8KtDLu4a-EM"]

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

OPENAI_API_KEY = "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"
loader = DirectoryLoader(path='./', glob = "**/*.txt", loader_cls=TextLoader,
                         show_progress=True)

index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).from_loaders([loader])

while True:
    print()
    question = input("Question: ")
    result = index.query(f'{question}', llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))
    print()
    print(result)