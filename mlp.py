from composer import ComposerModel, Trainer
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.callbacks import SpeedMonitor
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import os
import composer.utils.dist as composer_dist
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt

class FakeIterableDataset(IterableDataset):
    def __init__(self, n_samples, msl, d_model):
        self.n_samples = n_samples
        self.msl = msl
        self.d_model = d_model

    def __iter__(self):
        for _ in range (self.n_samples):
            yield {'inputs': torch.rand((self.msl, self.d_model), dtype=torch.bfloat16), 'targets': torch.ones((self.msl, self.d_model), dtype=torch.bfloat16)}

    def __len__(self):
        return self.n_samples

class MLP(ComposerModel):
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.net = nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(n_layers)])

    def forward(self, batch):
        outputs = self.net(batch['inputs'])
        return outputs

    def loss(self, outputs, batch):
        targets = batch['targets']
        return torch.nn.functional.cross_entropy(outputs, targets)

    def flops_per_batch(self, batch):
        bs, msl = batch['inputs'].shape[0:2]
        return 6 * bs * msl * self.n_layers * (self.d_model**2)

def main():
    n_samples = 1000
    msl = 2048

    global_train_batch_size = 32
    device_train_batch_size = global_train_batch_size // composer_dist.get_world_size()
    device_train_microbatch_size = 1
    print (global_train_batch_size, device_train_batch_size, device_train_microbatch_size)
    n_layers = 5
    d_model = 2048


    dataset = FakeIterableDataset(n_samples, msl, d_model)
    loader = DataLoader(
        dataset,
        batch_size=device_train_batch_size,
        drop_last=True,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = MLP(n_layers, d_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup='10ba', alpha_f=0.1)
    max_duration = '20ba'
    callbacks = [
        SpeedMonitor(window_size=5),
    ]

    trainer = Trainer(
        run_name='mlp',
        device='neuron',
        seed=42,
        model=model,
        train_dataloader=loader,
        eval_dataloader=None,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=0,
        progress_bar=False,
        log_to_console=True,
        console_log_interval='1ba',
        loggers=None,
        callbacks=callbacks,
        precision='amp_bf16',
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=None,
        save_folder=None,
        save_interval=None,
        save_num_checkpoints_to_keep=-1,
        python_log_level='debug',
        dist_timeout=300,
    )
    trainer.fit()

if __name__ == '__main__':
    main()
