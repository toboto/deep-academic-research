# Deep Academic Research

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Deep Academic Research is an academic research assistance system based on RAG (Retrieval-Augmented Generation) technology, focusing on generating high-quality academic review articles. By combining vector databases and large language models, the system can perform in-depth analysis and summarization of specific research topics or researchers' academic achievements.

## 🚀 Core Features

- **Research Topic Review**: Based on the `OverviewRAG` class, provides comprehensive analysis of specific research topics, generating complete reviews including background, theoretical foundations, methodologies, key findings, and emerging trends.
- **Researcher Achievement Review**: Based on the `PersonalRAG` class, performs in-depth analysis of specific researchers' academic achievements, generating personalized reviews including academic background, research evolution, core contributions, and academic impact.
- **Vector Database Support**: Uses Milvus vector database for storing and retrieving academic literature, supporting efficient semantic search.
- **Multilingual Support**: Supports bilingual output in Chinese and English, with accurate translation of technical terms through the `AcademicTranslator` class.

## 📖 Quick Start

### Requirements

- Python >= 3.10
- Milvus Vector Database
- MySQL Database (for storing academic literature metadata)

### Installation

```bash
# Clone repository
git clone https://github.com/toboto/deep-academic-research.git

# Create and activate virtual environment
cd deep-academic-research
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Configuration

1. Copy configuration template:
```bash
cp config.yaml.example config.yaml
```

2. Modify configuration file with necessary parameters:
- LLM API keys
- Vector database connection information
- MySQL database connection information
- Other optional configurations

### Usage Examples

#### Generate Research Topic Review

```python
from deepsearcher.agent.overview_rag import OverviewRAG
from deepsearcher.configuration import Configuration

# Initialize configuration
config = Configuration()
config.load_config("config.yaml")

# Create OverviewRAG instance
overview_rag = OverviewRAG(
    llm=config.get_llm(),
    reasoning_llm=config.get_reasoning_llm(),
    writing_llm=config.get_writing_llm(),
    translator=config.get_translator(),
    embedding_model=config.get_embedding_model(),
    vector_db=config.get_vector_db(),
    rbase_settings=config.get_rbase_settings()
)

# Generate review
result = overview_rag.query("AI Applications in Healthcare")
```

#### Generate Researcher Achievement Review

```python
from deepsearcher.agent.persoanl_rag import PersonalRAG

# Create PersonalRAG instance
personal_rag = PersonalRAG(
    llm=config.get_llm(),
    reasoning_llm=config.get_reasoning_llm(),
    writing_llm=config.get_writing_llm(),
    translator=config.get_translator(),
    embedding_model=config.get_embedding_model(),
    vector_db=config.get_vector_db(),
    rbase_settings=config.get_rbase_settings()
)

# Generate review
result = personal_rag.query("Please write a research overview of Professor Zhang San")
```

## 🔧 System Architecture

### Core Components

1. **OverviewRAG**: Research Topic Review Generator
   - Multi-chapter structure support
   - Automatic search query generation
   - Intelligent content optimization
   - Automatic abstract and conclusion generation

2. **PersonalRAG**: Researcher Achievement Review Generator
   - Researcher profile analysis
   - Research evolution tracking
   - Core contribution analysis
   - Academic impact assessment

3. **AcademicTranslator**: Academic Translator
   - Accurate technical term translation
   - Chinese-English bidirectional support
   - Terminology management

4. **Milvus Vector Database**: Knowledge Retrieval Engine
   - Efficient semantic search
   - Multi-dimensional filtering
   - Scalable storage architecture

### Data Flow

1. Data Preprocessing
   - Literature metadata extraction
   - Text chunking
   - Vectorization

2. Knowledge Retrieval
   - Semantic search
   - Relevance reordering
   - Result deduplication

3. Content Generation
   - Chapter content generation
   - Content optimization
   - Multilingual translation

## 🤝 Contributing

We welcome various forms of contributions, including but not limited to:

- Submitting issues and suggestions
- Improving documentation
- Submitting code improvements
- Sharing usage experiences

Please refer to [Contributing Guide](./CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](./LICENSE.txt) file for details.

## 🙏 Acknowledgments

This project is developed based on the [DeepSearcher](https://github.com/zilliztech/deep-searcher) project. We would like to express our gratitude to the original project's contributors.
