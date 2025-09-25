from llm import get_ai_message
import streamlit as st
from dotenv import load_dotenv


st.set_page_config(page_title="Tax-Chatbot", page_icon="📜")

st.title("📜 소득세 챗봇")
st.caption("소득세 관련 질문을 내용을 입력하면 자동으로 답변을 생성해줍니다.")

load_dotenv()

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
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
        
print(f"after === {st.session_state.message_list}")

