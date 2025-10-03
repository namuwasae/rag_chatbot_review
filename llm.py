from config import answer_examples
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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import FewShotChatMessagePromptTemplate


### Statefully manage chat history ###
store = {}

# 함수 이름 : get_session_history
# 함수 기능 : session_id를 입력받아 채팅 히스토리를 관리하는 함수
# 함수 파라미터 : session_id
# 함수 반환값 : BaseChatMessageHistory
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

# 함수 이름 : get_history_retriever
# 함수 기능 : 채팅 히스토리를 고려한 리트리버를 생성하는 함수. 이를 이용해 rag_chain을 만드는데 이용할 것임.
# 함수 파라미터 : 없음
# 함수 반환값 : history_aware_retriever
def get_history_retriever():
    retriever = get_retriever()
    llm = get_llm()

# 새 qa_chain. 시스템 프롬프트를 이용해 새로운 채팅 프롬프트를 만들 것임.
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

    return history_aware_retriever


# 함수 이름 : get_qa_chain
# 함수 기능 : RetrievalQA 체인 생성하는 함수
# 함수 파라미터 : 없음
# 함수 반환값 : qa_chain
# 함수 설명 : create_retrieval_chain과 create_document_chain을 이용해 rag_chain을 얻을 것임. 
def get_rag_chain():
    
    llm=get_llm()
    
    ### 이 부분이 지금 langchain 사이트에 없음. 지금 공개된 프롬프트도 확인해보고 그걸 바탕으로도 해봐도 될듯.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt, # 사실 few-shot prompt도 chat history임. 그래서 qa_prompt에 넣어줄거임.
        examples = answer_examples,
    )
    ###
    
    
    system_prompt = (
    "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
    "\n\n"
    "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, # 우리는 그동안 이 프롬프트대로 채팅을 해왔다고 입력.
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    history_aware_retriever = get_history_retriever()
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # rag_chain을 받아서 conversational_rag_chain을 만듦. 이제 채팅히스토리까지 포함된 리트리버인  conversational_rag_chain을 리턴해서 사용하면 됨.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') # pick('answer')를 해야 답변이 깔끔하게 answer파트만 나옴.

    return conversational_rag_chain

# chat_history를 얹어줘야 하는데, 딕셔너리를 하나 선언해서 메모리에 저장해 관리하는 방법이 일반적임. 딕셔너리를 이용하는 방식이니 이 방식은 앱이 종료되면 데이터가 날아감.

# 함수 이름 : get_ai_response
# 함수 기능 : user_message를 입력으로 받아 ai_response를 반환하는 함수
# 함수 파라미터 : user_message
def get_ai_response(user_message):

    dictionary_chain = get_dictionary()
    rag_chain = get_rag_chain()
    # conversational_rag_chain에서는 input_messages_key가 "input"임. 그리고 output_message_key가 "answer"이므로 question->answer로 바꿔야함.
    tax_chain = {"input":dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        }  # constructs a key "abc123" in `store`.
    )
# 여기서 tax_chain.invoke()["answer"]를 하면 스트리밍 할 때 에러날 가능성이 높음.

    return ai_response
        # return ai_message["answer"]

