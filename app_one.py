



#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_openai.embeddings import OpenAIEmbeddings

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

import chainlit as cl


from langchain.schema.runnable.config import RunnableConfig
from langchain.retrievers import MultiQueryRetriever



model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)


client = QdrantClient(path="Qdrant_db")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

collection_name = "Meta info 400"
qdrant =  Qdrant(client, collection_name, embedding_model)

qdrant_retriever = qdrant.as_retriever()
advanced_retriever = MultiQueryRetriever.from_llm(retriever=qdrant_retriever, llm=model)



@cl.on_chat_start
async def on_chat_start():

    RAG_PROMPT = """

    CONTEXT:
    {context}

    QUERY:
    {question}

    Answer the query above using the context provided. If you don't know the answer responde with: I don't know
    """

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    
    runnable = (
    {"context": itemgetter("question") | advanced_retriever, "question": itemgetter("question")} | rag_prompt | model | StrOutputParser()
    )


    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):

    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()