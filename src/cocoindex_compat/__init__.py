
import sys
import logging
from typing import Any, Callable, List, Dict, Generic, TypeVar

logger = logging.getLogger("cocoindex_mock")

T = TypeVar('T')

# --- Core Types ---
Json = Dict[str, Any]

class DataSlice(Generic[T]):
    def __init__(self, data=None):
        self.data = data
    
    def __class_getitem__(cls, item):
        return cls

    def transform(self, op, **kwargs):
        # In a mock, we might just return self or applying simple logic
        # For now, just return a new DataSlice to chain calls
        return DataSlice()
    
    def row(self):
        return RowContext()

class RowContext:
    def __enter__(self):
        return {} # Mock row dict
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class FlowBuilder:
    def add_source(self, source, **kwargs):
        return DataSlice()

class DataScope(dict):
    def add_collector(self):
        return Collector()

class Collector:
    def collect(self, **kwargs):
        pass
    def export(self, *args, **kwargs):
        pass

# --- LLM Support (Mock) ---
class LlmApiType:
    OPENAI = "openai"
    OLLAMA = "ollama"

class LlmSpec:
    def __init__(self, api_type, model):
        self.api_type = api_type
        self.model = model

# --- Decorators ---

def transform_flow(**kwargs):
    def decorator(func):
        class FlowWrapper:
            def __init__(self, fn):
                self.fn = fn
            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)
            def eval(self, input_val):
                # Mock implementation for text_to_embedding.eval
                # We need to return a vector. 
                # Let's use sentence-transformers if available, else random
                try:
                    import os
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["HF_HUB_ENABLE_INTERNET"] = "0"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
                    return model.encode(input_val).tolist()
                except:
                    return [0.1] * 384
            def eval_async(self, input_val):
                import asyncio
                f = asyncio.Future()
                f.set_result(self.eval(input_val))
                return f
        return FlowWrapper(func)
    return decorator

def flow_def(name=""):
    def decorator(func):
        return func
    return decorator

# --- Sources ---

class sources:
    class LocalFile:
        def __init__(self, path):
            self.path = path

# --- Functions ---

class functions:
    class SplitRecursively:
        def __init__(self, **kwargs): pass
    class SentenceTransformerEmbed:
        def __init__(self, model=""): pass
    class ExtractByLlm:
        def __init__(self, llm_spec, output_type, instruction):
            self.llm_spec = llm_spec
            self.output_type = output_type
            self.instruction = instruction

# --- Targets ---

class targets:
    class lancedb:
        class LanceDB:
            def __init__(self, db_uri, table_name): pass

# --- Enums/Constants ---

class GeneratedField:
    UUID = "uuid"

class VectorSimilarityMetric:
    COSINE_SIMILARITY = "cosine"
    L2_DISTANCE = "l2"

class VectorIndexDef:
    def __init__(self, field_name, metric): pass

# --- Init ---

def init():
    logger.info("CocoIndex Compat (Mock) Initialized")

# --- Export ---
# This file acts as the package
