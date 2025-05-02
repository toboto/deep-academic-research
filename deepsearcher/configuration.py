import os
from typing import Literal

import yaml

from deepsearcher.agent import ChainOfRAG, DeepSearch, NaiveRAG
from deepsearcher.agent.academic_translator import AcademicTranslator
from deepsearcher.agent.overview_rag import OverviewRAG
from deepsearcher.agent.rag_router import RAGRouter
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.loader.file_loader.base import BaseLoader
from deepsearcher.loader.web_crawler.base import BaseCrawler
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_YAML_PATH = os.path.join(current_dir, "..", "config.yaml")

FeatureType = Literal[
    "llm", "embedding", "file_loader", "web_crawler", "vector_db", "reasoning_llm", "writing_llm"
]


class Configuration:
    def __init__(self, config_path: str = DEFAULT_CONFIG_YAML_PATH):
        # Initialize default configurations
        config_data = self.load_config_from_yaml(config_path)
        self.provide_settings = config_data["provide_settings"]
        self.query_settings = config_data["query_settings"]
        self.load_settings = config_data["load_settings"]
        self.rbase_settings = config_data.get("rbase_settings", {})

    def load_config_from_yaml(self, config_path: str):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def set_provider_config(self, feature: FeatureType, provider: str, provider_configs: dict):
        """
        Set the provider and its configurations for a given feature.

        :param feature: The feature to configure (e.g., 'llm', 'file_loader', 'web_crawler').
        :param provider: The provider name (e.g., 'openai', 'deepseek').
        :param provider_configs: A dictionary with configurations specific to the provider.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        self.provide_settings[feature]["provider"] = provider
        self.provide_settings[feature]["config"] = provider_configs

    def get_provider_config(self, feature: FeatureType):
        """
        Get the current provider and configuration for a given feature.

        :param feature: The feature to retrieve (e.g., 'llm', 'file_loader', 'web_crawler').
        :return: A dictionary with provider and its configurations.
        """
        if feature not in self.provide_settings:
            raise ValueError(f"Unsupported feature: {feature}")

        return self.provide_settings[feature]


class ModuleFactory:
    def __init__(self, config: Configuration):
        self.config = config

    def _create_module_instance(self, feature: FeatureType, module_name: str):
        # e.g.
        # feature = "file_loader"
        # module_name = "deepsearcher.loader.file_loader"
        class_name = self.config.provide_settings[feature]["provider"]
        module = __import__(module_name, fromlist=[class_name])
        class_ = getattr(module, class_name)
        return class_(**self.config.provide_settings[feature]["config"])

    def create_llm(self) -> BaseLLM:
        return self._create_module_instance("llm", "deepsearcher.llm")

    def create_reasoning_llm(self) -> BaseLLM:
        return self._create_module_instance("reasoning_llm", "deepsearcher.llm")

    def create_writing_llm(self) -> BaseLLM:
        return self._create_module_instance("writing_llm", "deepsearcher.llm")

    def create_embedding(self) -> BaseEmbedding:
        return self._create_module_instance("embedding", "deepsearcher.embedding")

    def create_file_loader(self) -> BaseLoader:
        return self._create_module_instance("file_loader", "deepsearcher.loader.file_loader")

    def create_web_crawler(self) -> BaseCrawler:
        return self._create_module_instance("web_crawler", "deepsearcher.loader.web_crawler")

    def create_vector_db(self) -> BaseVectorDB:
        return self._create_module_instance("vector_db", "deepsearcher.vector_db")


config = Configuration()

module_factory: ModuleFactory = None
llm: BaseLLM = None
reasoning_llm: BaseLLM = None
writing_llm: BaseLLM = None
embedding_model: BaseEmbedding = None
file_loader: BaseLoader = None
vector_db: BaseVectorDB = None
web_crawler: BaseCrawler = None
default_searcher: RAGRouter = None
naive_rag: NaiveRAG = None
academic_translator: AcademicTranslator = None
overview_rag: OverviewRAG = None


def init_config(config: Configuration):
    global \
        module_factory, \
        llm, \
        reasoning_llm, \
        writing_llm, \
        embedding_model, \
        file_loader, \
        vector_db, \
        web_crawler, \
        default_searcher, \
        naive_rag, \
        academic_translator, \
        overview_rag
    module_factory = ModuleFactory(config)
    llm = module_factory.create_llm()

    # Initialize reasoning and writing models if they are configured
    log.debug("initializing reasoning_llm")
    if "reasoning_llm" in config.provide_settings:
        reasoning_llm = module_factory.create_reasoning_llm()
    else:
        reasoning_llm = llm  # Fallback to the default LLM if not configured

    log.debug("initializing writing_llm")
    if "writing_llm" in config.provide_settings:
        writing_llm = module_factory.create_writing_llm()
    else:
        writing_llm = llm  # Fallback to the default LLM if not configured

    log.debug("initializing embedding_model")
    embedding_model = module_factory.create_embedding()

    log.debug("initializing file_loader")
    file_loader = module_factory.create_file_loader()

    log.debug("initializing web_crawler")
    web_crawler = module_factory.create_web_crawler()

    log.debug("initializing vector_db")
    vector_db = module_factory.create_vector_db()

    log.debug("initializing default_searcher")
    default_searcher = RAGRouter(
        llm=llm,
        rag_agents=[
            DeepSearch(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
            ChainOfRAG(
                llm=llm,
                embedding_model=embedding_model,
                vector_db=vector_db,
                max_iter=config.query_settings["max_iter"],
                route_collection=True,
                text_window_splitter=True,
            ),
        ],
    )
    naive_rag = NaiveRAG(
        llm=llm,
        embedding_model=embedding_model,
        vector_db=vector_db,
        top_k=10,
        route_collection=True,
        text_window_splitter=True,
    )

    log.debug("initializing academic_translator")
    # Initialize AcademicTranslator
    academic_translator = AcademicTranslator(llm=llm, rbase_settings=config.rbase_settings)

    log.debug("initializing overview_rag")
    # Initialize OverviewRAG
    try:
        # Initialize OverviewRAG
        overview_rag = OverviewRAG(
            llm=llm,
            reasoning_llm=reasoning_llm,
            writing_llm=writing_llm,
            translator=academic_translator,
            embedding_model=embedding_model,
            vector_db=vector_db,
            text_window_splitter=config.rbase_settings.get("overview_rag", {}).get(
                "text_window_splitter", True
            ),
            rbase_settings=config.rbase_settings,
        )

        log.info("OverviewRAG initialized successfully")
    except Exception as e:
        log.critical(f"Failed to initialize OverviewRAG: {e}")
        overview_rag = None
