# Retrieval-Augmented Generation: A Short Overview

## Abstract
Retrieval-Augmented Generation (RAG) combines a parametric language model with a
non-parametric memory in the form of a searchable document index. Instead of relying
solely on knowledge stored in model weights, a RAG system retrieves relevant passages
at query time and conditions generation on them, producing answers that are more
factual and easier to attribute to sources.

## Motivation
Large language models hallucinate: they produce fluent but factually incorrect text.
They are also frozen at their training cutoff and cannot cite sources. RAG addresses
both problems by grounding generation in retrieved evidence.

## Architecture
A typical RAG pipeline has five stages:

1. Ingestion and chunking of documents into passages.
2. Embedding of passages into dense vectors.
3. Indexing of vectors in a vector database such as FAISS or ChromaDB.
4. Retrieval of the top-k most similar passages for a query.
5. Generation of an answer conditioned on the retrieved passages.

## Evaluation
RAG systems are evaluated along two axes: retrieval quality (are the right passages
retrieved?) and generation quality (is the answer faithful to the retrieved context?).
The RAGAS framework provides metrics including faithfulness, answer relevancy, context
precision, and context recall.

## Conclusion
RAG is a practical pattern for building trustworthy question-answering systems over
private or rapidly changing document collections.
