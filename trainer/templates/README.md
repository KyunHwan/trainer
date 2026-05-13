# Templates

This folder contains the `Protocol` definitions that establish contracts for every pluggable component. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Define the `Trainer`, `DatasetFactory`, `LossFactory`, and `OptimizerFactory` protocols
- Serve as reference skeletons for implementing custom components
- Enable runtime type checking via `@runtime_checkable`

## Layout

| File | Protocol | Method signatures |
|------|----------|-------------------|
| [`trainer.py`](trainer.py) | `Trainer` | `__init__(models: nn.ModuleDict, optimizers: dict, loss: nn.Module)` and `train_step(data: dict, stats: Any) -> dict[str, Any]` |
| [`dataset.py`](dataset.py) | `DatasetFactory` | `build(**kwargs) -> dict[str, Any]` |
| [`loss.py`](loss.py) | `LossFactory` | `build() -> nn.Module` |
| [`optim.py`](optim.py) | `OptimizerFactory` | `build(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer` |

## Contracts

- All four protocols are `@runtime_checkable`. The framework verifies conformance with `isinstance(obj, Protocol)` after construction (currently only for `Trainer`, in `_build_trainer`).
- `isinstance` against a runtime-checkable protocol checks method *names*, not signatures. A mismatch in parameters is only caught at call time.
- **`Trainer.train_step` canonical call is `trainer.train_step(data=data, stats=stats_gpu)`** — these are the kwargs the framework actually passes from the loop. The protocol declaration in [`trainer.py`](trainer.py) reflects this signature.
- **`Trainer.__init__` also receives `device`** in practice, even though the protocol declaration omits it. `instantiate` in [`utils/import_utils.py`](../utils/import_utils.py) filters constructor kwargs by signature; `device=...` is forwarded only when the constructor accepts it.
- `DatasetFactory.build` receives `opt_params={'local_rank', 'dist_enabled', 'save_dir'}` and `params=<datamodule.params>` from the trainer. Return `{"dataset": Dataset, "norm_stats": dict}` (or just the dataset; the loop tolerates both).
- `OptimizerFactory.build` receives a model's `parameters()` iterator. The factory is responsible for any scheduler integration; the loop calls only `optimizer.step()`.

## How to extend

See [docs/07_extending.md](../../docs/07_extending.md) for a recipe per protocol.

## Cross-links

- Protocol contracts: [docs/04_concepts.md § Protocols (templates)](../../docs/04_concepts.md#protocols-templates)
- Trainer contract: [docs/04_concepts.md § The train_step contract](../../docs/04_concepts.md#the-train_step-contract)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- The `Trainer.train_step()` return dict is logged directly to wandb on rank 0. Tensor values must be scalar (the logger calls `.detach().item()`); non-scalar tensors raise.
- `LossFactory.build()` returning an `nn.Module` causes the framework to call `.to(device)` on it. If you return something else, that move is skipped — check that your custom-loss object knows what device it's on.
- The framework's `isinstance(obj, Protocol)` check happens *after* construction. A trainer class that lacks `train_step` will instantiate successfully and then fail the isinstance check with `TypeError: Constructed object does not match Trainer interface`.
