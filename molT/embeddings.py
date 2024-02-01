import torch
import torch.nn as nn
from rdkit.Chem.rdchem import ChiralType, HybridizationType, StereoType

from .config import MolTConfig
from .utils import TokenType, unpack_atom_properties, unpack_bond_properties


class AtomPropertyEmbedder(nn.Module):
    def __init__(self, config: MolTConfig) -> None:
        super().__init__()
        self.config = config
        self.in_ring_embedding = nn.Embedding(
            3, config.embedding_size, padding_idx=0
        )
        self.charge_embedding = nn.Embedding(
            4, config.embedding_size, padding_idx=0
        )
        self.hybridization_embedding = nn.Embedding(
            len(HybridizationType.values) + 1,
            config.embedding_size,
            padding_idx=0,
        )
        self.chirality_embedding = nn.Embedding(
            len(ChiralType.values) + 1, config.embedding_size, padding_idx=0
        )

    def sparse_embedding_select(self, sparse_prop, prop_embedding):
        sparse_prop = sparse_prop.coalesce().long()
        B, L = sparse_prop.shape
        D = self.config.embedding_size
        I, V = sparse_prop.indices(), sparse_prop.values()
        sparse_embeds = torch.sparse_coo_tensor(I, prop_embedding(V), size=(B, L, D))
        return sparse_embeds

    def forward(
        self,
        prop_atom_in_ring,
        prop_atom_charge,
        prop_atom_hybridization,
        prop_atom_chirality,
    ):
        in_ring_embeds = self.sparse_embedding_select(prop_atom_in_ring, self.in_ring_embedding)
        charge_embedding = self.sparse_embedding_select(prop_atom_charge, self.charge_embedding)
        hybridization_embedding = self.sparse_embedding_select(
            prop_atom_hybridization, self.hybridization_embedding
        )
        chirality_embedding = self.sparse_embedding_select(prop_atom_chirality, self.chirality_embedding)

        prop_embedding = (
            in_ring_embeds
            + charge_embedding
            + hybridization_embedding
            + chirality_embedding
        )

        return prop_embedding


class BondPropertyEmbedder(nn.Module):
    def __init__(self, config: MolTConfig) -> None:
        super().__init__()
        self.config = config
        self.aromatic_embedding = nn.Embedding(
            3, config.embedding_size, padding_idx=0
        )

        self.conjugated_embedding = nn.Embedding(
            3, config.embedding_size, padding_idx=0
        )

        self.stereo_embedding = nn.Embedding(
            len(StereoType.values) + 1, config.embedding_size, padding_idx=0
        )

    def sparse_embedding_select(self, sparse_prop, prop_embedding):
        sparse_prop = sparse_prop.coalesce().long()
        B, L = sparse_prop.shape
        D = self.config.embedding_size
        I, V = sparse_prop.indices(), sparse_prop.values()
        sparse_embeds = torch.sparse_coo_tensor(I, prop_embedding(V), size=(B, L, D))
        return sparse_embeds

    def forward(
        self,
        prop_bond_aromatic,
        prop_bond_conjugated,
        prop_bond_stereo,
    ):
        aromatic_embeds = self.sparse_embedding_select(prop_bond_aromatic, self.aromatic_embedding)
        conjugated_embeds = self.sparse_embedding_select(prop_bond_conjugated, self.conjugated_embedding)
        stereo_embeds = self.sparse_embedding_select(prop_bond_stereo, self.stereo_embedding)

        prop_embedding = aromatic_embeds + conjugated_embeds + stereo_embeds
        return prop_embedding


class MolFeatureEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, mol_feat_mask, mol_features):
        mol_feature_embeddings = torch.where(
            mol_feat_mask.unsqueeze(-1),
            input_embeddings * mol_features.unsqueeze(-1),
            0.0,
        )
        return mol_feature_embeddings

@torch.no_grad()
def scale_target(x):
    return x


class RegressionTargetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, target_mask, target_values):
        target_values = scale_target(target_values)
        target_encoded_embeddings = torch.where(
            target_mask.unsqueeze(-1),
            input_embeddings * target_values.unsqueeze(-1),
            0.0,
        )
        return target_encoded_embeddings


def batched_select(batch_idx, batch_lp_embeds):
    return batch_lp_embeds[batch_idx]


class PositionEmbedder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vectorized_batch_select = torch.vmap(batched_select)

    @torch.no_grad()
    def forward(self, pos_embed_ids, lp_embeds, token_type_ids):
        lp_embeds = torch.nan_to_num(lp_embeds, 0.0)
        # Not sure how to construct a sparse pos_embedding tensor
        # using the values from pos_embed_idxs to index from lp_embeds
        # makes CUDA crash, possibly due to OOM
        pos_embed_ids = pos_embed_ids.to_dense().long()

        if self.training:
            B, L, _ = lp_embeds.shape
            random_signs = -1 + 2 * torch.randint(
                0, 2, (B, L), device=lp_embeds.device
            ).unsqueeze(-1)
            lp_embeds = lp_embeds * random_signs

        pos_embeds = self.vectorized_batch_select(pos_embed_ids, lp_embeds)
        pos_embeds = pos_embeds.flatten(-2, -1)

        return pos_embeds


class MolTEmbeddings(nn.Module):
    def __init__(self, config: MolTConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.pos_embeddings = PositionEmbedder(config)
        self.atom_prop_embeddings = AtomPropertyEmbedder(config)
        self.bond_prop_embeddings = BondPropertyEmbedder(config)

        if self.config.use_mol_descriptor_tokens:
            self.mol_feature_embeddings = MolFeatureEmbedder()

        if self.config.use_target_token:
            self.target_embedding = RegressionTargetEmbedder()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

        self.type_embeddings = nn.Embedding(len(TokenType), config.embedding_size)

    def forward(
        self,
        input_ids,
        token_type_ids,
        pos_embed_ids,
        lp_embeds,
        atom_props,
        bond_props,
        mol_features,
        target_values,
        **kwargs,
    ):
        pos_embeds = self.pos_embeddings(pos_embed_ids, lp_embeds, token_type_ids)
        token_type_embeds = self.type_embeddings(token_type_ids)
        input_embeddings = self.embeddings(input_ids)

        if self.config.use_mol_descriptor_tokens:
            mol_feat_mask = token_type_ids == TokenType.FEAT
            input_embeddings += self.mol_feature_embeddings(
                input_embeddings, mol_feat_mask, mol_features
            )

        if self.config.use_target_token:
            target_mask = token_type_ids == TokenType.TGT
            input_embeddings += self.target_embedding(
                input_embeddings, target_mask, target_values
            )

        atom_props = unpack_atom_properties(atom_props)
        bond_props = unpack_bond_properties(bond_props)
        atom_prop_embeddings = self.atom_prop_embeddings(**atom_props)
        bond_prop_embeddings = self.bond_prop_embeddings(**bond_props)

        prop_embeddings = atom_prop_embeddings + bond_prop_embeddings

        embeddings = torch.cat(
            [input_embeddings + token_type_embeds + prop_embeddings, pos_embeds], dim=-1
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
