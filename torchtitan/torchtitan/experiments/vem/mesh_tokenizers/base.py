from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Tuple
import numpy as np
import trimesh
import torch

    
class LogitsProcessor:
    def process_logits(self, logits: torch.Tensor, batch: torch.Tensor):
        '''
        logits: N x C
        batch: N x L
        '''
        return logits


class TokenizerSpec(ABC):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        tokenizer_options (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def tokenize(self, text: str) -> np.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        pass

    def detokenize(self, ids: np.ndarray) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text

        Raises:
            NotImplementedError: Non-abstract, optional method
        """
        raise NotImplementedError("{} has no method 'detokenize'".format(type(self).__name__))

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Convert embedding ids to text offsets

        Args:
            ids (list[int]): The ids to convert
            text (str): The text to convert

        Returns:
            list[int]: The converted offsets

        Raises:
            NotImplementedError: Non-abstract, optional method
        """
        raise NotImplementedError("{} has no method 'offsets'".format(type(self).__name__))

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token"""
        pass

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token"""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """The vocabulary size"""
        pass

    @property
    def cls(self):
        """The CLS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'cls'".format(type(self).__name__))

    @property
    def sep(self):
        """The SEP token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'sep'".format(type(self).__name__))

    @property
    def pad(self):
        """The PAD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))

    @property
    def eod(self):
        """The EOD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'eod'".format(type(self).__name__))

    @property
    def bos(self):
        """The BOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        """The EOS token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))

    @property
    def mask(self):
        """The MASK token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        raise NotImplementedError("{} has no attribute 'mask'".format(type(self).__name__))
    
    def get_logits_processor(self) -> LogitsProcessor:
        """Get the logits processor for this tokenizer

        Returns:
            LogitsProcessor: The logits processor for this tokenizer
        """
        return None