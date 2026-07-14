import torch
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.tools.logging import init_logger, logger


def clamp(x, min_val, max_val):
    return max(min(x, max_val), min_val)

class ShardedEMA(Stateful):
    """
    Exponential Moving Average for FSDP2 models (local shards only).
    Works with torch.distributed.checkpoint.save/load the same as FSDP models.
    """

    def __init__(
        self, 
        model, 
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 1.0,
        min_value = 0.0,

        device = None, 
        use_num_updates = True
    ):
        self.beta = beta
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.device = device
        self.use_num_updates = use_num_updates
        self.step = 0
        self.rank = torch.distributed.get_rank()

        # initialize shadow params with shard-local tensors
        self.shadow_params = {}
        self.name_mapping = {}
        for name, p in model.named_parameters():
            canonical_name = self._remove_field(name, target="_orig_mod")
            canonical_name = self._remove_field(canonical_name, target="_checkpoint_wrapped_module")
            self.name_mapping[name] = canonical_name
            if p.requires_grad:
                self.shadow_params[canonical_name] = p.detach().clone().to(device or p.device)
                # print(name, type(self.shadow_params[name]), self.shadow_params[name].to_local().shape, self.shadow_params[name].shape, self.shadow_params[name].placements)

    def get_current_decay(self):
        epoch = max(self.step - self.update_after_step - 1, 0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** (- self.power)

        # return value.clamp(min = self.min_value, max = self.beta).item()
        return clamp(value, self.min_value, self.beta)
    
    @torch.no_grad()
    def copy_from(self, model):
        """Copy EMA params from the model (shard-local)."""
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            canonical_name = self.name_mapping[name]
            self.shadow_params[canonical_name].copy_(p.detach().to(self.shadow_params[canonical_name].device))
    
    def _remove_field(self, name, target='_orig_mod'):
        fields = [i for i in name.split('.') if i != target]
        return '.'.join(fields)

    @torch.no_grad()
    def update(self, model):
        """Update EMA params with shard-local parameters."""
        self.step += 1
        if self.step < self.update_after_step:
            logger.info("Skipping EMA update until step {} (currently at {})".format(self.update_after_step, self.step))
        elif self.step == self.update_after_step:
            logger.info("Initializing EMA parameters at step {}".format(self.step))
            self.copy_from(model)
        elif self.step > self.update_after_step and self.step % self.update_every == 0:
            d = self.get_current_decay()
            logger.info("EMA update at step {} with decay {}".format(self.step, d))
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                canonical_name = self.name_mapping[name]
                sp = self.shadow_params[canonical_name]
                sp.mul_(d).add_(p.detach().to(sp.device), alpha=1 - d)

    def state_dict(self):
        """Return EMA state in shard-local form (checkpointable)."""
        return {
            "model": self.shadow_params,
            "step": self.step,
        }

    def load_state_dict(self, state):
        """Load EMA state (must match local shard structure)."""
        self.step = state["step"]
        logger.info(f"Loading EMA state at rank {self.rank}, step={self.step}")
        for k, v in state["model"].items():
            self.shadow_params[k].copy_(v)