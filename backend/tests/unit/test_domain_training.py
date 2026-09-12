import pytest
import torch
from pathlib import Path

from app.domain_training.config import DomainTrainingConfig
from app.domain_training.dataset import DomainTrainingDataset
from app.domain_training.sampler import MixedDataset
from app.domain_training.checkpoints import DomainCheckpointManager
from app.domain_training.trainer import DomainTrainer

def test_domain_training_config():
    config = DomainTrainingConfig()
    assert config.learning_rate < 1e-3  # Lower LR for domain fine-tuning
    assert config.domain_ratio + config.general_conversation_ratio + config.instruction_ratio == pytest.approx(1.0)

def test_domain_checkpoint_manager(tmp_path):
    mgr = DomainCheckpointManager(run_dir=tmp_path)
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    config = DomainTrainingConfig()
    
    ckpt_path = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=10,
        epoch=1,
        metrics={"loss": 0.5},
        config=config,
        name="latest"
    )
    
    assert ckpt_path.exists()
    
    step, epoch, metrics = mgr.load_checkpoint(ckpt_path, model, optimizer)
    assert step == 10
    assert epoch == 1

def test_domain_trainer_tiny_overfit(tmp_path):
    config = DomainTrainingConfig()
    config.output_dir = str(tmp_path / "runs")
    
    trainer = DomainTrainer(config=config)
    results = trainer.run_overfit_test(num_steps=15)
    
    assert results["passed"] is True
    assert results["final_loss"] < results["initial_loss"]
