import os
import argparse
from datetime import datetime
import datasets
import transformers

import deeph3


def train_model(dataset_dir, model_dir, batch_size=64, epochs=5):
    dataset = datasets.load_from_disk(dataset_dir)

    tokenizer = transformers.BertTokenizerFast(
        vocab_file="deeph3/models/AntiBERTy/vocab.txt", do_lower_case=False)
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    config = transformers.BertConfig(vocab_size=len(tokenizer.vocab),
                                     hidden_size=128,
                                     max_position_embeddings=512,
                                     num_attention_heads=8,
                                     num_hidden_layers=4)
    model = transformers.BertForMaskedLM(config=config)

    training_args = transformers.TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(model_dir)


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))

    desc = 'Creates Huggingface dataset from OAS paired csv files'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='The directory containing HF dataset for training')
    now = str(datetime.now().strftime('%y-%m-%d %H:%M:%S'))
    default_model_path = os.path.join(
        project_path, 'trained_models/BERTpaired_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs

    train_model(dataset_dir, output_dir, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    cli()
