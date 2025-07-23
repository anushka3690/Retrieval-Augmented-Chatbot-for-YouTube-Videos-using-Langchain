# Retrieval-Augmented Chatbot for YouTube Videos using LangChain

A chatbot that can answer questions about YouTube video content by extracting transcripts, creating embeddings, and using retrieval-augmented generation (RAG) with open-source language models.

## Features

- **YouTube Transcript Extraction**: Automatically fetch transcripts from YouTube videos
- **Document Processing**: Split transcripts into manageable chunks for better processing
- **Vector Search**: Create embeddings and store in FAISS vector database for efficient retrieval
- **RAG Pipeline**: Combine retrieval with generation for accurate, context-aware responses
- **Open-Source Models**: Uses HuggingFace models for both embeddings and text generation
- **Interactive Chain**: Built with LangChain for modular and extensible pipeline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Retrieval-Augmented-Chatbot-for-YouTube-Videos-using-Langchain.git
cd Retrieval-Augmented-Chatbot-for-YouTube-Videos-using-Langchain
```

2. Install required dependencies:
```bash
pip install youtube-transcript-api langchain-community langchain-openai \
            faiss-cpu tiktoken python-dotenv sentence-transformers \
            transformers torch langchain-huggingface
```

## Usage

### Basic Setup

1. **Import Required Libraries**:
```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
```

2. **Extract YouTube Transcript**:
```python
video_id = "YOUR_VIDEO_ID"  # Extract from YouTube URL
transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
transcript = " ".join(chunk.text for chunk in transcript_list)
```

3. **Process and Store Documents**:
```python
# Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(transcript)

# Create embeddings and vector store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = [Document(page_content=chunk) for chunk in chunks]
vectorstore = FAISS.from_documents(documents, embedding)
```

4. **Set up Retrieval and Generation**:
```python
# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Initialize language model
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100)
)
```

5. **Ask Questions**:
```python
# Build the complete RAG chain
main_chain = parallel_chain | prompt | llm | parser
response = main_chain.invoke("What is the main topic of this video?")
```

### Example Questions

- "Can you summarize the video?"
- "What are the key points discussed?"
- "Is [specific topic] mentioned in the video?"
- "Who are the main speakers or people mentioned?"

## Technical Architecture

### Components

1. **Document Ingestion**: 
   - YouTube transcript extraction using `youtube-transcript-api`
   - Error handling for videos without captions

2. **Text Processing**:
   - `RecursiveCharacterTextSplitter` for optimal chunk sizing
   - Configurable chunk size (1000) and overlap (200)

3. **Embedding & Vector Store**:
   - HuggingFace `all-MiniLM-L6-v2` for sentence embeddings
   - FAISS for efficient similarity search

4. **Retrieval**:
   - Similarity-based retrieval with configurable top-k results
   - Context formatting for prompt construction

5. **Generation**:
   - TinyLlama model for lightweight text generation
   - Customizable temperature and token limits

6. **Pipeline**:
   - LangChain's `RunnableParallel` for parallel processing
   - Modular chain construction with `|` operator

### Pipeline Flow

```
YouTube Video → Transcript Extraction → Text Splitting → Embedding Generation → 
Vector Store → Question → Retrieval → Context + Question → LLM → Answer
```

## Configuration

### Model Settings

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Language Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval Top-K**: 4 documents

### Environment Variables

Set your HuggingFace cache directory:
```python
os.environ['HF_HOME'] = 'your/cache/directory'
```

## Limitations

- Requires videos to have available transcripts/captions
- English language support (can be extended)
- Context window limitations of the language model
- Accuracy depends on transcript quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [HuggingFace](https://huggingface.co/) for open-source models
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

