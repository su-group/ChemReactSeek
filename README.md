# Knowledge-Guided Chemical Reaction Protocol Design with Augmented Large Language Models: Structured Data Extraction and Contextual Knowledge Integration

### *PLZ NOT download or share before the paper is published*



## Overview

We present ChemReactSeek, a unified platform combining automated data extraction, dynamic knowledge base integration, and semantic vectorization to enable mechanism-informed, data-driven experimental protocol design for complex chemical reactions.

------------
## Installation

- Python 3.9+
- fitz（PyMuPDF）
- sentence_transformers
- faiss
------------
## Usage
**Extraction_Structured_Data**
*Utilizing a large language model API to extract structured information on reaction procedures from a collection of PDF literature.*
Please modify the code at the corresponding location according to the provided instructions.

*The extracted data feeds a dynamically scalable domain-specific knowledge repository, capturing mechanistic insights and empirical constraints for protocol generation.*

**Contextual_Knowledge_Integration**
*Based on the knowledge repository, the application of a large language model API enables tailored responses to domain-specific scientific inquiries and facilitates the design of experimental protocols.*

Here, we directly use the training model all - MiniLM - L6 - v2, you can download it from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 for the model.
Please modify the code at the corresponding location according to the provided instructions.

------------
## License
The implementation code and associated data for this research are publicly accessible and distributed under the terms of the MIT License (https://opensource.org/licenses/MIT).
