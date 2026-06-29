# CS_KG_RAG-System
 
A conversational AI assistant that answers questions by combining structured knowledge graph retrieval with semantic vector search — built as an extension of my MSc thesis at Vrije Universiteit Amsterdam.
 
---
 
## What this project does
 
Most RAG systems retrieve from unstructured text. This one retrieves from a **structured knowledge graph** built from the Visual Genome dataset, where every fact carries a confidence score and a typicality score derived from how frequently it appeared across the dataset.
 
When a question comes in, the system runs two retrievals in parallel:
 
- **SPARQL graph traversal** over a curated RDF-star knowledge graph loaded in GraphDB
- **Semantic vector search** over the same facts embedded with sentence-transformers in ChromaDB
The results are merged using a confidence-weighted ranking, and passed to Gemini 2.5 Flash to generate a grounded response. Every answer cites the supporting triples along with their statistical metadata, so the output is traceable back to the knowledge graph.
 
---
 
## How the knowledge graph was built
 
The KG is the output of my master's thesis. In short:
 
- Extracted ~7 million subject–predicate–object triples from Visual Genome scene and region graphs
- Normalised predicates and mapped all terms to WordNet synsets for semantic consistency
- Filtered down to **14,290 high-confidence triples** using Wilson lower-bound scoring and a minimum support threshold of k=3
- Serialised to RDF-star Turtle format with statistical annotations (`:confidence`, `:typicality`, `:support`)
- Loaded into GraphDB for SPARQL querying
The result is a compact, semantically clean knowledge graph focused on everyday physical objects, spatial relations, and human activities.
 
---
 
## Tech stack
 
| Component | Tool |
|---|---|
| Knowledge graph | GraphDB, RDF-star, SPARQL |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers |
| LLM | Gemini 2.5 Flash via Google AI Studio |
| Orchestration | LangChain |
| Interface | Streamlit |
| Deployment | Docker |
 
---
 
## Project structure
 
```
CS_KG_RAG-System/
├── data/               # Knowledge graph triples and source data
├── src/                # Core pipeline: retrieval, merging, generation
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```
 
---
 
## Project status
 
This is a portfolio and research project built to demonstrate hybrid neuro-symbolic retrieval. The knowledge graph and pipeline are fully functional but require a local GraphDB instance and a Gemini API key to run.
 
---
 
## Background
 
This project grew out of my MSc thesis:
 
> *Constructing and Analyzing a Commonsense Knowledge Graph from Visual Genome*
> Vrije Universiteit Amsterdam, August 2025

The thesis focused on building and evaluating the knowledge graph. This project asks the next question: once you have a reliable, confidence-annotated KG, how do you make it useful for a conversational AI system?
 
---
 
## Author
 
**Divyansh Singh**
MSc Artificial Intelligence, Vrije Universiteit Amsterdam
