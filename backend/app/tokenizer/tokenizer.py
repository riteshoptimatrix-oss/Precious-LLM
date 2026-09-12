"""
Precious Edu LLM — Custom BPE Tokenizer Facade

Main high-level Tokenizer class providing clean encode, decode, batch, save, and load methods.
Decoupled from FastAPI, MongoDB, or PyTorch Transformer logic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import BPEEncoder
from app.tokenizer.decoder import BPEDecoder
from app.tokenizer.formatter import ConversationFormatter
from app.tokenizer.serialization import TokenizerSerializer

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Custom BPE Tokenizer Facade.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        merges: List[Tuple[str, str]],
        config: Optional[TokenizerConfig] = None
    ):
        self.config = config or get_tokenizer_config()
        self.vocab = vocab
        self.merges = merges

        self.encoder = BPEEncoder(self.vocab, self.merges, self.config)
        self.decoder = BPEDecoder(self.vocab, self.config)
        self.formatter = ConversationFormatter(self.config)
        self.serializer = TokenizerSerializer()

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text string into integer token IDs."""
        return self.encoder.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode integer token IDs back into reconstructed string."""
        return self.decoder.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode a list of text strings into lists of token IDs."""
        return [self.encode(txt, add_bos=add_bos, add_eos=add_eos) for txt in texts]

    def save(
        self,
        output_dir: Path,
        dataset_version: str = "0.1.0",
        statistics: Optional[Dict] = None
    ) -> Path:
        """Save tokenizer artifacts to directory."""
        return self.serializer.save(
            vocab=self.vocab,
            merges=self.merges,
            output_dir=output_dir,
            config=self.config,
            dataset_version=dataset_version,
            statistics=statistics
        )

    @classmethod
    def load(cls, input_dir: Path, config: Optional[TokenizerConfig] = None) -> "Tokenizer":
        """Load frozen tokenizer from artifact directory."""
        serializer = TokenizerSerializer()
        vocab, merges, _ = serializer.load(Path(input_dir))
        return cls(vocab=vocab, merges=merges, config=config)
