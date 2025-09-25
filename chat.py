import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
# query -> 직장인 -> 거주자로 바꾸는 chain을 추가를 위한 라이브러리
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone


st.set_page_config(page_title="Tax-Chatbot", page_icon="📜")

st.title("📜 소득세 챗봇")
st.caption("소득세 관련 질문을 내용을 입력하면 자동으로 답변을 생성해줍니다.")

load_dotenv()

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 함수 이름 : get_ai_message
# 함수 기능 : user_message를 입력으로 받아 ai_message를 반환하는 함수
# 함수 파라미터 : user_message
def get_ai_message(user_message):
    # Pinecone 클라이언트 초기화 및 인덱스 가져오기
    pc = Pinecone()
    index_name = 'tax-markdown-index'
    index = pc.Index(index_name)
    
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    database = PineconeVectorStore(index=index, embedding=embeddings)
    
    # RAG 프롬프트와 LLM 설정
    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
    
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=database.as_retriever(search_kwargs={"k":4}),
        chain_type_kwargs={"prompt": rag_prompt}
    )

    # query -> 직장인 -> 거주자로 바꾸는 chain을 추가
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    dictionary_prompt = ChatPromptTemplate.from_template(f"""
                    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
                    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
                    사전 : {dictionary}
                    질문 : {{question}}
                    """)

    dictionary_chain = dictionary_prompt | llm | StrOutputParser() 

    tax_chain = {"query":dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message

if user_question := st.chat_input(placeholder="소득세 관련 질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    
    with st.spinner("답변 생성중입니다."):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
        
print(f"after === {st.session_state.message_list}")

