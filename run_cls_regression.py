import os
from functools import partial

import pandas as pd
from astartes import train_test_split
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SchedulerType

import datasets as hds
from data import generate_and_scale_mol_descriptors
from molT import (
    CLSRegression,
    DataCollatorForMaskedMolecularModeling,
    MolTConfig,
    MolTTokenizer,
)
from utils import download_model_from_wandb

os.environ["WANDB_PROJECT"] = "molt_ablation"


def tokenize(entry, tokenizer):
    entry = dict(entry)
    smiles = entry.pop("smiles")
    return tokenizer(
        smiles,
        truncation=False,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        **entry,
    )


def train_func(model, ds, data_collator):
    training_args = TrainingArguments(
        output_dir="molT_runs",
        evaluation_strategy="steps",
        learning_rate=1e-4,
        num_train_epochs=32,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=4,
        eval_steps=8,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        report_to="wandb",
        dataloader_num_workers=16,
        lr_scheduler_type=SchedulerType.COSINE,
        data_seed=42,
        run_name="molt_cls",
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        bf16=True,
        bf16_full_eval=True,
        max_grad_norm=0.5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=model.report_metrics,
    )

    trainer.train()


def load_gsk_dataset():
    df = pd.read_csv("./datasets/gsk_processed.csv")
    splits = train_test_split(
        X=df["smiles"].to_numpy(),
        y=df["per_inh"].to_numpy(),
        sampler="scaffold",
        random_state=42,
        return_indices=True,
    )

    train_ids, val_ids = splits[-2], splits[-1]
    df_train = df.iloc[train_ids]
    df_val = df.iloc[val_ids]

    return hds.DatasetDict(
        {
            "train": hds.Dataset.from_pandas(df_train, preserve_index=False),
            "test": hds.Dataset.from_pandas(df_val, preserve_index=False),
        }
    )


if __name__ == "__main__":
    model_config = MolTConfig(target_col_name="per_inh",
        atom_bond_mask_probability=0.0,
        molecule_feature_mask_probability=0.0,
        target_mask_probability=0.0,
        use_target_token=False
    )
    tokenizer = MolTTokenizer(model_config)

    model_dir = download_model_from_wandb("rahul-e-dev", "molt", "molt_400K_8EP", "v0")
    model = CLSRegression.from_pretrained(model_dir, config=model_config)

    ds = load_gsk_dataset()

    ds, _ = generate_and_scale_mol_descriptors(
        ds, model_config.mol_descriptors, num_samples=5000, num_proc=16
    )

    tok_func = partial(tokenize, tokenizer=tokenizer)
    ds = ds.map(tok_func, num_proc=16)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForMaskedMolecularModeling(
        tokenizer=tokenizer, config=model_config
    )

    train_func(model, ds, data_collator)
