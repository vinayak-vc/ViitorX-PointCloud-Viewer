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


def cap_refine_pause(strategy) -> bool:
    """Nerfstudio 1.1.5 Splatfacto bug on large datasets: it builds gsplat's DefaultStrategy with
    pause_refine_after_reset = <training images> + refine_every, and gsplat only grows/prunes when
    `step % reset_every >= pause_refine_after_reset`. reset_every is 30 x 100 = 3000 steps, so with
    more than ~2,900 training images that test is never true: no Gaussian is ever split, cloned or
    pruned, while the opacity resets still fade everything every 3000 steps. Seen on a 10,849-image
    job: 108,004 seed Gaussians unchanged for 40k steps, 85% left near-transparent, PSNR 16.9.

    Cap the pause at half the reset cycle (1500 steps by default), so refinement runs in the second
    half of every cycle - closer to the original 3DGS schedule, which has no pause at all. Small
    datasets (pause already below the cap) are untouched. Returns True when the value was changed."""
    reset = getattr(strategy, "reset_every", 0) or 0
    pause = getattr(strategy, "pause_refine_after_reset", 0) or 0
    cap = reset // 2
    if reset <= 0 or pause <= cap:
        return False
    strategy.pause_refine_after_reset = cap
    print("pf360: Splatfacto refinement pause capped %d -> %d steps (reset every %d) - otherwise "
          "Gaussians are never densified or pruned with this many images" % (pause, cap, reset), flush=True)
    return True


def _patch_splatfacto() -> None:
    try:
        from nerfstudio.models.splatfacto import SplatfactoModel
    except Exception:                       # other Nerfstudio versions / methods: nothing to patch
        return
    original = SplatfactoModel.populate_modules

    def populate_modules(self, *a, **kw):
        original(self, *a, **kw)
        strategy = getattr(self, "strategy", None)
        if strategy is not None:
            cap_refine_pause(strategy)

    SplatfactoModel.populate_modules = populate_modules


def _patch_timer() -> None:
    """Nerfstudio's TimeWriter times with time.time(), which on Windows can step in ~16 ms ticks.
    The trainer then divides by that duration (rays/sec after the one-image eval every 100
    steps), so a fast eval measures 0.0 s and training dies with ZeroDivisionError - seen at
    step ~900 of a splatfacto-big run. perf_counter is high-resolution; nerfstudio.utils.writer
    uses `time` only for TimeWriter's start/stop, so swapping that module's clock is safe."""
    try:
        from nerfstudio.utils import writer
    except Exception:
        return
    if hasattr(writer, "time"):
        writer.time = time.perf_counter


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
    _patch_splatfacto()
    _patch_timer()
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
