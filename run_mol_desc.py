import os
from functools import partial

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SchedulerType

from data import generate_and_scale_mol_descriptors
from molT import (
    DataCollatorForMaskedMolecularModeling,
    MolTConfig,
    MolTForMaskedMM,
    MolTTokenizer,
)

from utils import download_model_from_wandb

os.environ["WANDB_PROJECT"] = "molt"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "all"


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
        num_train_epochs=8,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=8,
        eval_steps=32,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        report_to="wandb",
        dataloader_num_workers=16,
        lr_scheduler_type=SchedulerType.COSINE,
        data_seed=42,
        run_name="molt_ph2",
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        max_grad_norm=0.5,
        save_steps=128,
        save_strategy="steps",
        load_best_model_at_end=True,
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


if __name__ == "__main__":
    model_config = MolTConfig(
        atom_bond_mask_probability=0.15,
        molecule_feature_mask_probability=0.15,
        use_mol_descriptor_tokens=True,
        use_target_token=False,
    )
    tokenizer = MolTTokenizer(model_config)

    model_dir = download_model_from_wandb("rahul-e-dev", "molt", "molt_ph1", "v0")
    model = MolTForMaskedMM.from_pretrained(model_dir, config=model_config)
    
    ds = (
        load_dataset("sagawa/ZINC-canonicalized")["validation"]
        .select(range(500_000))
        .train_test_split(seed=42)
    )

    ds, _ = generate_and_scale_mol_descriptors(
        ds, model_config.mol_descriptors, num_samples=250_000, num_proc=32
    )

    tok_func = partial(tokenize, tokenizer=tokenizer)
    ds = ds.map(tok_func, num_proc=32)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForMaskedMolecularModeling(
        tokenizer=tokenizer, config=model_config
    )

    train_func(model, ds, data_collator)
