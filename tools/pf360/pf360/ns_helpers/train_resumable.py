"""Runs INSIDE the Nerfstudio Python environment: `ns-train` with a graceful pause.

usage: python train_resumable.py <pause_file> <ns-train arguments...>

Identical to `ns-train <arguments>`, plus: when <pause_file> appears, the
trainer's own stop flag is set, so it finishes the current step, saves a
checkpoint (Trainer._after_train -> save_checkpoint) and exits with code 75.
pf360 then resumes later with `--load-checkpoint` (restore point). A hard kill
instead would lose everything since the last periodic checkpoint.

Exit codes: 0 = training finished, 75 = paused with a fresh checkpoint,
anything else = Nerfstudio's own failure.
"""
import sys
import threading
import time
from pathlib import Path

EXIT_PAUSED = 75


def main() -> int:
    pause = Path(sys.argv[1])
    try:
        pause.unlink()                     # a stale request from an earlier run
    except FileNotFoundError:
        pass
    from nerfstudio.engine import trainer as trainer_mod
    from nerfstudio.scripts.train import entrypoint

    state = {"paused": False}
    original_train = trainer_mod.Trainer.train

    def train(self):
        done = threading.Event()

        def watch():
            while not done.wait(0.5):
                if pause.exists():
                    state["paused"] = True
                    print("pf360: pause requested - saving a checkpoint at the current step", flush=True)
                    self.stop_training = True      # loop breaks; _after_train saves the checkpoint
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            original_train(self)
        finally:
            done.set()

    trainer_mod.Trainer.train = train

    # Nerfstudio 1.1.5 + Splatfacto resume bug: Trainer.setup() builds the optimizers, THEN
    # loads the checkpoint, and SplatfactoModel.load_state_dict replaces every gauss_params
    # tensor with a new Parameter sized for the checkpoint's Gaussian count. The optimizers
    # keep pointing at the old (initial-size) tensors, and the loaded Adam state is keyed by
    # them, so the first densification step indexes mismatched arrays (CUDA device-side
    # assert in gsplat strategy ops.split). Re-point each optimizer, with its state, at the
    # current parameter.
    original_load = trainer_mod.Trainer._load_checkpoint

    def load_checkpoint(self):
        original_load(self)
        pipeline = getattr(self.pipeline, "module", self.pipeline)
        params = getattr(pipeline.model, "gauss_params", None)
        if params is None:
            return
        rebound = 0
        for name, opt in self.optimizers.optimizers.items():
            if name not in params or len(opt.param_groups) != 1 or len(opt.param_groups[0]["params"]) != 1:
                continue
            old, new = opt.param_groups[0]["params"][0], params[name]
            if old is new:
                continue
            state = opt.state.pop(old, None)
            opt.param_groups[0]["params"][0] = new
            if state is not None:
                opt.state[new] = state
            rebound += 1
        if rebound:
            print("pf360: re-bound %d optimizer(s) to the resumed Gaussians (%d)"
                  % (rebound, params["means"].shape[0]), flush=True)

    trainer_mod.Trainer._load_checkpoint = load_checkpoint
    sys.argv = ["ns-train"] + sys.argv[2:]
    entrypoint()
    if state["paused"]:
        try:
            pause.unlink()
        except FileNotFoundError:
            pass
        time.sleep(0.2)
        return EXIT_PAUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
