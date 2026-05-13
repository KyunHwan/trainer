# optimizer

This folder contains optimizer factory implementations registered in `OPTIMIZER_BUILDER_REGISTRY`. For project-wide context, see [docs/README.md](../../../docs/README.md).

## Purpose

- Provide optimizer + learning rate schedule combinations for training
- Integrate scheduler stepping inside the optimizer (no separate scheduler step needed in the training loop)
- Support checkpoint save/load of both optimizer and scheduler state

## Layout

| File | Registry key | Description |
|------|-------------|-------------|
| [`adamw_cosine_decay.py`](adamw_cosine_decay.py) | `adamw_warmup_cosine_decay` | `AdamWWithWarmupCosine` — AdamW with a linear warmup → cosine decay schedule. Scheduler steps inside `optimizer.step()`. Scheduler state lives under `"scheduler"` in the optimizer's `state_dict()` |
| [`adamw_onecyclelr.py`](adamw_onecyclelr.py) | `adamw_cosine_schedule` | `AdamWWithOneCycle` — AdamW with PyTorch's `OneCycleLR` schedule integrated. Same step-inside-step pattern |
| [`schedule_free_radam.py`](schedule_free_radam.py) | `schedule_free_radam` | Factory wrapping `schedulefree.RAdamScheduleFree` from the [schedulefree](https://github.com/facebookresearch/schedule_free) library |

## Contracts

Each factory implements `OptimizerFactory`:

```python
@OPTIMIZER_BUILDER_REGISTRY.register("key")
class MyOptimizerFactory:
    def __init__(self, ...): ...
    def build(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer: ...
```

At runtime, `_build_optimizers` iterates `model.component_optims`: for each entry, looks up the factory by `type`, instantiates from `params`, calls `factory.build(models[name].parameters())`. The result is keyed by component name in the optimizer dict passed to the trainer.

## How to extend

See [docs/07_extending.md § Recipe: a new optimizer](../../../docs/07_extending.md#recipe-a-new-optimizer).

If your optimizer bundles a scheduler:

- Step the scheduler inside the optimizer's `step()` override.
- Save the scheduler's `state_dict()` under the `"scheduler"` key inside the optimizer's `state_dict()`.
- Restore the scheduler state in `load_state_dict()` after stripping the key.

This pattern keeps the training loop free of scheduler bookkeeping — the loop only calls `optimizer.step()`.

## Cross-links

- Recipe: [docs/07_extending.md § Recipe: a new optimizer](../../../docs/07_extending.md#recipe-a-new-optimizer)
- Resume semantics: [docs/08_checkpoints_and_resume.md](../../../docs/08_checkpoints_and_resume.md)
- Hub: [docs/README.md](../../../docs/README.md)

## Gotchas / invariants

- Schedulers are stepped inside `optimizer.step()`, so the training loop must **not** step a scheduler separately.
- Optimizer state dicts include scheduler state under the `"scheduler"` key. `load_state_dict` restores both.
- One optimizer is created per model component listed in `component_optims`. Frozen models (no trainable params) are silently skipped — even if you list them in `component_optims`, no optimizer is built because `[p for p in models[name].parameters() if p.requires_grad]` is empty.
- `peak_lr` must be > 0 and `start_lr`/`end_lr` must be ≤ `peak_lr` for `adamw_warmup_cosine_decay`. Validation happens at construction time.
