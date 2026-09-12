"""
Precious Edu LLM — Mixed Dataset Sampler

Combines domain examples, general conversational examples, and instruction examples
according to configured ratios to prevent catastrophic forgetting.
"""

import logging
import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MixedDataset(Dataset):
    """
    Wrapper dataset mixing multiple source datasets according to ratio weights.
    """

    def __init__(
        self,
        domain_dataset: Dataset,
        general_dataset: Optional[Dataset] = None,
        instruction_dataset: Optional[Dataset] = None,
        domain_ratio: float = 0.5,
        general_ratio: float = 0.3,
        instruction_ratio: float = 0.2,
        seed: int = 42
    ):
        self.examples: List[Dict[str, torch.Tensor]] = []

        # 1. Add domain examples
        if len(domain_dataset) > 0:
            self.examples.extend([domain_dataset[i] for i in range(len(domain_dataset))])

        # 2. Add general dataset examples if provided
        if general_dataset and len(general_dataset) > 0:
            count = int(len(domain_dataset) * (general_ratio / domain_ratio)) if len(domain_dataset) > 0 else len(general_dataset)
            gen_indices = [i % len(general_dataset) for i in range(count)]
            self.examples.extend([general_dataset[i] for i in gen_indices])

        # 3. Add instruction dataset examples if provided
        if instruction_dataset and len(instruction_dataset) > 0:
            count = int(len(domain_dataset) * (instruction_ratio / domain_ratio)) if len(domain_dataset) > 0 else len(instruction_dataset)
            inst_indices = [i % len(instruction_dataset) for i in range(count)]
            self.examples.extend([instruction_dataset[i] for i in inst_indices])

        # Shuffle deterministically
        rng = random.Random(seed)
        rng.shuffle(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
