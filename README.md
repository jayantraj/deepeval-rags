# DeepEval-RAGS

## Project Overview

**DeepEval-RAGS** is a Retrieval-Augmented Generation (RAG) system that uses OpenAI's `text-embedding-3-small` model for embeddings and Pinecone for vector-based document retrieval. The system implements query re-ranking using LLMs to enhance result relevance. The core evaluation is performed using `deepeval`'s **AnswerRelevancyMetric**, ensuring that generated responses are relevant based on the retrieved context.

---

## Features
- **Embedding Model**: OpenAI `text-embedding-3-small` for creating text embeddings.
- **Vector Database**: Pinecone for storing and retrieving document embeddings.
- **LLM Re-ranking**: re-rank retrieved documents for better relevance.
- **Answer Relevancy Metric**: Evaluates whether the LLM output is relevant or useful, based on the retrieved context.


