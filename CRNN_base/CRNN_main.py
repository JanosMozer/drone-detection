import os
import pandas as pd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from CRNN_config import (
    DATA_CONFIG, TRAINING_CONFIG, TASK_CONFIG, WANDB_CONFIG, MODEL_CONFIG, AUDIO_CONFIG,
    CRNNModel, create_data_loaders
)


def main():
    # Load dataset
    df = pd.read_csv(DATA_CONFIG['csv_path'])
    
    # Create results directory
    os.makedirs(DATA_CONFIG['results_dir'], exist_ok=True)
    
    # Initialize wandb
    wandb_logger = WandbLogger(
        entity=WANDB_CONFIG['entity'],
        project=WANDB_CONFIG['project'],
        tags=WANDB_CONFIG['tags'],
        notes=WANDB_CONFIG['notes'],
        config={
            **AUDIO_CONFIG,
            **MODEL_CONFIG,
            **TRAINING_CONFIG,
            **TASK_CONFIG,
            'dataset_size': len(df),
            'class_distribution': df['class'].value_counts().to_dict()
        }
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(df)
    
    # Initialize model
    model = CRNNModel(
        task_name=TASK_CONFIG['task_name'],
        num_classes=TASK_CONFIG['num_classes'],
        loss_type=TASK_CONFIG['loss_type'],
        class_weights=TASK_CONFIG['class_weights']
    )
    
    
    # Setup progress bar theme
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="white",
            progress_bar="white",
            progress_bar_finished="white",
            batch_progress="white",
            time="white",
            processing_speed="white",
            metrics="white"
        ),
        leave=True
    )
    
    # Setup callbacks
    callbacks = [
        progress_bar,
        EarlyStopping(
            monitor=f"{TASK_CONFIG['task_name']}_val_loss", 
            patience=TRAINING_CONFIG['patience_early_stop'], 
            min_delta=TRAINING_CONFIG['min_delta']
        ),
        ModelCheckpoint(
            dirpath=DATA_CONFIG['results_dir'], 
            filename='best_aircraft_detection_model', 
            monitor=f"{TASK_CONFIG['task_name']}_val_loss", 
            save_top_k=1,
            mode='min'
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=TRAINING_CONFIG['max_epochs'],
        callbacks=callbacks,
        logger=wandb_logger,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=False  # We handle this manually
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
