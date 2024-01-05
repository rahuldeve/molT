import rdkit.Chem as Chem
from transformers import PretrainedConfig

descriptors = [
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "qed",
    "SPS",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "AvgIpc",
    "BalabanJ",
    "BertzCT",
]


class MolTConfig(PretrainedConfig):
    def __init__(
        self,
        laplace_embeds_size=12,
        embedding_size=384,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size_multiplier=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        atom_bond_mask_probability=0.15,
        molecule_feature_mask_probability=0.15,
        target_mask_probability=0.8,
        target_col_name="qed",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        pt = Chem.GetPeriodicTable()  # type: ignore
        self.atoms = [pt.GetElementSymbol(i) for i in range(1, 119)]
        self.bonds = list(Chem.rdchem.BondType.names.keys())
        # make sure target is not presented as a molecule descriptor
        self.mol_descriptors = descriptors
        self.num_special_tokens = 8

        self.vocab_size = (
            len(self.atoms)
            + len(self.bonds)
            + len(self.mol_descriptors)
            + self.num_special_tokens
        )

        self.laplace_embeds_size = laplace_embeds_size
        self.embedding_size = embedding_size
        self.hidden_size = 3 * self.embedding_size + 2 * self.laplace_embeds_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size_multiplier * self.hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.atom_bond_mask_probability = atom_bond_mask_probability
        self.molecule_feature_mask_probability = molecule_feature_mask_probability
        self.target_mask_probability = target_mask_probability
        self.target_col_name = target_col_name
        self.feature_names = list(set(descriptors) - set([target_col_name]))
