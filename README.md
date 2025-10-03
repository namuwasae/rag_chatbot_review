# 소득세 챗봇

LangChain과 Streamlit을 이용한 소득세 관련 질문 답변 챗봇입니다.

## 주요 기능

- 소득세 관련 질문에 대한 정확한 답변
- 채팅 히스토리 관리
- Few-shot 학습을 통한 답변 품질 향상
- 실시간 스트리밍 응답

## 기술 스택

- **Frontend**: Streamlit
- **Backend**: LangChain, OpenAI GPT-4o-mini
- **Vector Database**: Pinecone
- **Embedding**: OpenAI text-embedding-3-large

## 로컬 실행 방법

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경변수 설정:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# secrets.toml 파일에 실제 API 키 입력
```

3. 앱 실행:
```bash
streamlit run chat.py
```

## Streamlit Cloud 배포 방법

1. GitHub 저장소에 코드 푸시
2. Streamlit Cloud에서 새 앱 생성
3. 환경변수 설정:
   - `OPENAI_API_KEY`: OpenAI API 키
   - `PINECONE_API_KEY`: Pinecone API 키
4. 배포 시작

## 문제 해결

### 배포 오류 시 확인사항

1. **requirements.txt**: 필수 패키지만 포함되어 있는지 확인
2. **환경변수**: API 키가 올바르게 설정되어 있는지 확인
3. **Import 오류**: deprecated import가 수정되어 있는지 확인
4. **메모리 사용량**: Pinecone 인덱스 크기 확인

### 자주 발생하는 오류

- `installer returned a non-zero exit code`: requirements.txt에 불필요한 패키지 제거
- `ImportError`: LangChain 버전 호환성 확인
- `API Key Error`: 환경변수 설정 확인
