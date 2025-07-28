# Round 1B: Persona-Driven Document Intelligence - Approach Explanation

## Methodology Overview

My solution implements a semantic similarity-based approach to extract and rank document sections based on persona requirements and job-to-be-done specifications. The system processes multiple PDFs simultaneously and delivers contextually relevant content tailored to specific user needs.

## Core Components

### 1. Text Extraction and Section Identification
- **PDF Processing**: Uses PyPDF2 for robust text extraction across multiple document formats
- **Section Detection**: Implements heuristic-based section identification using pattern matching for headings (uppercase, title case, numbered sections)
- **Content Segmentation**: Splits documents into logical sections with titles and content blocks

### 2. Semantic Understanding
- **Embedding Model**: Leverages SentenceTransformer's 'all-MiniLM-L6-v2' model (~80MB) for efficient semantic embeddings
- **Query Construction**: Combines persona description with job-to-be-done to create targeted search queries
- **Similarity Scoring**: Uses cosine similarity between query embeddings and section embeddings for relevance ranking

### 3. Relevance Ranking System
- **Multi-document Analysis**: Processes 3-10 PDFs simultaneously as specified
- **Contextual Scoring**: Ranks sections based on semantic alignment with persona needs
- **Top-K Selection**: Returns the most relevant sections (configurable, default top-10)

### 4. Subsection Analysis
- **Granular Extraction**: Identifies key subsections within relevant sections
- **Sentence-level Analysis**: Uses NLTK for sentence tokenization and selection
- **Content Refinement**: Extracts diverse, informative sentences representing section essence

Architecture

Text Extraction: PyPDF2 for robust PDF processing
Semantic Analysis: SentenceTransformers for embedding-based similarity
Section Detection: Heuristic-based heading identification
Relevance Ranking: Cosine similarity scoring
Subsection Analysis: NLTK-based sentence extraction

Models and Libraries Used

SentenceTransformers: all-MiniLM-L6-v2 (~80MB) for semantic embeddings
PyPDF2: PDF text extraction
scikit-learn: Cosine similarity calculations
NLTK: Natural language processing and tokenization
NumPy: Numerical operations

Building and Running
Docker Build
docker build --platform linux/amd64 -t persona-analyzer:v1 .

Docker Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/out

