# loss

This folder contains loss function implementations registered in `LOSS_BUILDER_REGISTRY`. For project-wide context, see [docs/README.md](../../../docs/README.md).

## Purpose

- Provide loss functions for training imitation-learning policies
- Support standard losses (L2/MSE) and specialized losses (Sinkhorn-Knopp optimal transport)
- Provide utility loss functions for mixture-of-experts architectures

## Layout

| File | Registry key | Description |
|------|-------------|-------------|
| [`l2.py`](l2.py) | `l2_loss` | Factory wrapping `torch.nn.MSELoss` with configurable reduction |
| [`sinkhorn_knopp.py`](sinkhorn_knopp.py) | `sinkhorn_knopp` | Sinkhorn-Knopp optimal transport loss using `geomloss.SamplesLoss`. Computes OT distance between predicted and target (action, state) pairs with configurable state weighting |
| [`moe_gating_loss.py`](moe_gating_loss.py) | *(not registered)* | Utility functions for MoE auxiliary losses: `router_z_loss` (z-loss for router logits) and `switch_load_balancing_loss` (load balancing across experts) |

## Contracts

Each factory implements `LossFactory`:

```python
@LOSS_BUILDER_REGISTRY.register("key")
class MyLossFactory:
    def build(self) -> nn.Module: ...
```

At runtime, `_build_loss` looks the factory up by `train.loss.type`, instantiates it with `train.loss.params`, calls `.build()`, moves the resulting `nn.Module` to `device`, and passes it into the trainer as `loss=`. The trainer is responsible for actually calling the loss in its `train_step`.

## How to extend

See [docs/07_extending.md § Recipe: a new loss](../../../docs/07_extending.md#recipe-a-new-loss).

## Cross-links

- Recipe: [docs/07_extending.md § Recipe: a new loss](../../../docs/07_extending.md#recipe-a-new-loss)
- Hub: [docs/README.md](../../../docs/README.md)

## Gotchas / invariants

- The loss factory's `build()` returns an `nn.Module`, which is moved to the training device by `_build_loss()`. See [`offline_trainer.py`](../../../trainer/offline_trainer.py).
- `KOTSinkhornLoss` uses `math.sqrt(lam_state)` to scale state features so that the squared Euclidean distance in the concatenated space equals `||a-a'||^2 + lam_state * ||s-s'||^2`.
- The `sinkhorn_knopp` factory requires `geomloss` (installed by [`env_setup.sh`](../../../env_setup.sh)).
- `moe_gating_loss.py` is not registered — it provides standalone utility functions intended to be imported and called directly from trainer implementations that use mixture-of-experts.
