import torch
import torch.nn as nn
from rdkit.Chem.rdchem import ChiralType, HybridizationType, StereoType

from .config import MolTConfig
from .utils import TokenType, unpack_atom_properties, unpack_bond_properties


class AtomPropertyEmbedder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.per_prop_embedding_size = config.embedding_size // 4
        assert self.per_prop_embedding_size * 4 == config.embedding_size

        self.in_ring_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )
        self.charge_embedding = nn.Embedding(
            4, self.per_prop_embedding_size, padding_idx=0
        )
        self.hybridization_embedding = nn.Embedding(
            len(HybridizationType.values) + 1,
            self.per_prop_embedding_size,
            padding_idx=0,
        )
        self.chirality_embedding = nn.Embedding(
            len(ChiralType.values) + 1, self.per_prop_embedding_size, padding_idx=0
        )

    def forward(
        self,
        prop_atom_in_ring,
        prop_atom_charge,
        prop_atom_hybridization,
        prop_atom_chirality,
    ):
        in_ring_embeds = self.in_ring_embedding(prop_atom_in_ring.long())
        charge_embedding = self.charge_embedding(prop_atom_charge.long())
        hybridization_embedding = self.hybridization_embedding(
            prop_atom_hybridization.long()
        )
        chirality_embedding = self.chirality_embedding(prop_atom_chirality.long())

        prop_embedding = torch.cat(
            [
                in_ring_embeds,
                charge_embedding,
                hybridization_embedding,
                chirality_embedding,
            ],
            dim=-1,
        )

        return prop_embedding


class BondPropertyEmbedder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.per_prop_embedding_size = config.embedding_size // 3
        assert self.per_prop_embedding_size * 3 == config.embedding_size

        self.aromatic_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )

        self.conjugated_embedding = nn.Embedding(
            3, self.per_prop_embedding_size, padding_idx=0
        )

        self.stereo_embedding = nn.Embedding(
            len(StereoType.values) + 1, self.per_prop_embedding_size, padding_idx=0
        )

    def forward(
        self,
        prop_bond_aromatic,
        prop_bond_conjugated,
        prop_bond_stereo,
    ):
        aromatic_embeds = self.aromatic_embedding(prop_bond_aromatic.long())
        conjugated_embeds = self.conjugated_embedding(prop_bond_conjugated.long())
        stereo_embeds = self.stereo_embedding(prop_bond_stereo.long())

        prop_embedding = torch.cat(
            [
                aromatic_embeds,
                conjugated_embeds,
                stereo_embeds,
            ],
            dim=-1,
        )
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


class RegressionTargetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, target_mask, target_values):
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
        B, L = token_type_ids.shape
        pos_embed_ids = pos_embed_ids.reshape((B, L, -1)).long()
        lp_embeds = lp_embeds.reshape((B, L, -1))

        if self.training:
            random_signs = -1 + 2 * torch.randint(
                0, 2, (B, L), device=lp_embeds.device
            ).unsqueeze(-1)
            lp_embeds = lp_embeds * random_signs

        pos_embeds = self.vectorized_batch_select(pos_embed_ids, lp_embeds)
        pos_embeds = pos_embeds.flatten(-2, -1)

        atom_mask = token_type_ids == TokenType.ATOM
        bond_mask = token_type_ids == TokenType.BOND
        atom_bond_mask = atom_mask | bond_mask
        pos_embeds = torch.where(atom_bond_mask.unsqueeze(-1), pos_embeds, 0.0)
        return pos_embeds


class MolTEmbeddings(nn.Module):
    def __init__(self, config: MolTConfig):
        super().__init__()
        self.embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.pos_embeddings = PositionEmbedder(config)
        self.atom_prop_embeddings = AtomPropertyEmbedder(config)
        self.bond_prop_embeddings = BondPropertyEmbedder(config)
        self.mol_feature_embeddings = MolFeatureEmbedder()
        # self.target_embedding = RegressionTargetEmbedder()

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

        mol_feat_mask = token_type_ids == TokenType.FEAT
        input_embeddings += self.mol_feature_embeddings(
            input_embeddings, mol_feat_mask, mol_features
        )

        # target_mask = token_type_ids == TokenType.TGT
        # input_embeddings += self.target_embedding(
        #     input_embeddings, target_mask, target_values
        # )

        atom_props = unpack_atom_properties(atom_props)
        bond_props = unpack_bond_properties(bond_props)
        atom_prop_embeddings = self.atom_prop_embeddings(**atom_props)
        bond_prop_embeddings = self.bond_prop_embeddings(**bond_props)

        prop_embeddings = atom_prop_embeddings + bond_prop_embeddings

        embeddings = torch.cat(
            [input_embeddings, pos_embeds, token_type_embeds, prop_embeddings], dim=-1
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
