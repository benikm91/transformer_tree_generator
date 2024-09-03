from dataclasses import dataclass

from experiments.tree.tree_structure import SampleGen


@dataclass
class ModelConfig:
    model: str
    n_layer: int
    n_head: int
    n_embd: int
    weight_init: bool = True
    positional_encoding: str = 'global_learn'
    tie_embeddings: bool = False
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0

    def __post_init__(self):
        # Some sanity checks
        assert self.positional_encoding in ['global_learn', 'global_sinusoidal']
        assert self.model in ['decoder', 'decoder-split', 'decoder-split-min']
        assert self.model != 'decoder-split-min' or self.branch_factor is not None
        assert 0 <= self.attn_pdrop < 1
        assert 0 <= self.embd_pdrop < 1
        assert 0 < self.n_layer and 0 < self.n_head and 0 < self.n_embd

    def to_file_name(self):
        return f"{self.model}_l{self.n_layer}_h{self.n_head}_e{self.n_embd}_wi{int(self.weight_init)}_pe{self.positional_encoding}_te{int(self.tie_embeddings)}_ap{self.attn_pdrop:.2f}_ep{self.embd_pdrop:.2f}"


@dataclass
class ExperimentConfig:
    model_config: ModelConfig
    sample_gen: SampleGen
    batch_size: int
    num_runs: int = 1
    curriculum: bool = False

    def __post_init__(self):
        self.batch_size = self.batch_size

    def to_file_name(self):
        return f"{self.model_config.to_file_name()}__{self.sample_gen.full_name()}__{self.batch_size}"

    @property
    def model(self) -> str:
        return self.model_config.model
