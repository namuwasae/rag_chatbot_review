from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
# query -> 직장인 -> 거주자로 바꾸는 chain을 추가를 위한 라이브러리
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pinecone import Pinecone
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
from langchain.schema.runnable import RunnableWithMessageHistory
from langchain.schema.messages import BaseChatMessageHistory


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



# 함수 이름 : get_llm
# 함수 기능 : LLM 초기화하는 함수
# 함수 파라미터 : model
# 함수 반환값 : llm
def get_llm(model = "gpt-4o-mini"):
    llm = ChatOpenAI(model = model, temperature = 0)
    return llm


# 함수 이름 : get_dictionary
# 함수 기능 : 자주 쓰이는 단어인 거주자로 사람 관련 표현을 매핑해주는 함수
# 함수 파라미터 : 없음
# 함수 반환값 : dictionary_chain
def get_dictionary():
    # query -> 직장인 -> 거주자로 바꾸는 chain을 추가
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
                    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
                    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
                    사전 : {dictionary}
                    질문 : {{question}}
                    """)

    dictionary_chain = dictionary_prompt | llm | StrOutputParser() 
    return dictionary_chain


# 함수 이름 : get_retriever
# 함수 기능 : 질문을 임베딩해 벡터로 만들고 이를 벡터스토어에 래핑하고 유사도를 비교해 유사도가 높은 상위 k개의 문서 청크를 가져오는 함수
# 함수 파라미터 : 없음
# 함수 반환값 : retriever
def get_retriever():
    # Pinecone 클라이언트 초기화 및 인덱스 가져오기
    pc = Pinecone()
    index_name = 'tax-markdown-index'
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    database = PineconeVectorStore(index=index, embedding=embeddings)
    retriever=database.as_retriever(search_kwargs={"k":4})
    return retriever

# 함수 이름 : get_qa_chain
# 함수 기능 : RetrievalQA 체인 생성하는 함수
# 함수 파라미터 : 없음
# 함수 반환값 : qa_chain
def get_rag_chain():
    
    llm=get_llm()
    retriever = get_retriever()

# 시스템 프롬프트를 이용해 새로운 채팅 프롬프트를 만들 것임.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # 이 history_aware_retriever를 이용해서 
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # rag_chain을 받아서 conversational_rag_chain을 만듦
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain



    # # RetrievalQA 체인 생성
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": rag_prompt}
    # )


# 함수 이름 : get_ai_message
# 함수 기능 : user_message를 입력으로 받아 ai_message를 반환하는 함수
# 함수 파라미터 : user_message
def get_ai_message(user_message):
    # 필요한 인자들인 llm, dictionary_chain, retriever, qa_chain을 가져옴
    llm = get_llm()
    dictionary_chain = get_dictionary()
    retriever = get_retriever()
    qa_chain = get_rag_chain()

    tax_chain = {"query":dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message["result"]

