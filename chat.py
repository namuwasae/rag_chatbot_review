from llm import get_ai_response
import streamlit as st
import os


st.set_page_config(page_title="Tax-Chatbot", page_icon="📜")

st.title("📜 소득세 챗봇")
st.caption("소득세 관련 질문을 내용을 입력하면 자동으로 답변을 생성해줍니다.")

# 환경변수 설정 (Streamlit secrets 또는 환경변수 사용)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])



if user_question := st.chat_input(placeholder="소득세 관련 질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    
    with st.spinner("답변 생성중입니다."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
            # streamlit에서는 채팅을 칠때마다 UI를 다시그림. 그래서 세션리스트에는 최종으로 나온 전체 답변을 넣어줘야 다음 채팅이 들어왔을 때 에러가 안남.
        
print(f"after === {st.session_state.message_list}")

