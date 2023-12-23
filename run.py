import os

from datasets import load_dataset
from sklearn import metrics
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import SchedulerType

from data import generate_and_scale_mol_descriptors
from molT import (
    DataCollatorForMaskedMolecularModeling,
    MolTConfig,
    MolTForMaskedMM,
    MolTTokenizer,
)

os.environ["WANDB_PROJECT"] = "molt"


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


def fn_metrics(eval_pred):
    (
        mm_loss,
        atom_prop_loss,
        bond_prop_loss,
        mol_desc_loss,
        target_loss,
        target_mask,
        pred_target_values,
        true_target_values,
    ) = eval_pred.predictions

    target_mask[target_mask == -100] = 0
    target_mask = target_mask.astype(bool)
    pred_target_values[~target_mask] = 0.0
    true_target_values[~target_mask] = 0.0

    y_true = true_target_values[target_mask]
    y_pred = pred_target_values[target_mask]

    r2_score = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    return {
        "mm_loss": mm_loss.mean(),
        "atom_prop_loss": atom_prop_loss.mean(),
        "bond_prop_loss": bond_prop_loss.mean(),
        "mol_desc_loss": mol_desc_loss.mean(),
        "target_loss": target_loss.mean(),
        "target_r2": r2_score,
        "target_mae": mae,
        "target_mse": mse,
    }


def train_func(model, ds, data_collator):
    training_args = TrainingArguments(
        output_dir="molT_runs",
        evaluation_strategy="steps",
        learning_rate=2e-4,
        num_train_epochs=4,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=8,
        eval_steps=32,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
        report_to="wandb",
        dataloader_num_workers=16,
        lr_scheduler_type=SchedulerType.COSINE,
        data_seed=42,
        run_name="molt_dev_v6",
        dataloader_pin_memory=True,
        bf16=True,
        bf16_full_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=fn_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    model_config = MolTConfig()
    tokenizer = MolTTokenizer(model_config)
    model = MolTForMaskedMM(model_config)

    ds = (
        load_dataset("sagawa/ZINC-canonicalized")["validation"]
        .select(range(500_000))
        .train_test_split(seed=42)
    )

    ds, _ = generate_and_scale_mol_descriptors(
        ds, model_config.mol_descriptors, num_samples=250_000, num_proc=32
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForMaskedMolecularModeling(
        tokenizer=tokenizer, mlm_probability=model_config.mlm_probability
    )

    train_func(model, ds, data_collator)
