"""Training length in Steps (the Training tab's Epochs / Steps pills): the form offers the control, its state counts the
steps an epoch makes from the training folder, the launch passes the step target only in Steps (never for a
fine-tune), and the trainer accepts it."""
import os

from fizgig.families import form, launch
from fizgig.families.registry import training_families


def _desc():
    return next(f for f in training_families() if f.key == "krea2")


def _folder(tmp_path, n):
    for i in range(n):
        (tmp_path / f"p{i}.png").write_bytes(b"")
    (tmp_path / "p0-masklabel.png").write_bytes(b"")        # a mask is not a training picture
    return str(tmp_path)


def test_form_has_one_length_control():
    rows = [r for s in form.form(_desc())["sections"] for r in s["rows"] if r]
    row = next(r for r in rows if r["key"] == "MAX_TRAIN_EPOCHS")
    assert row["kind"] == "length"
    assert row["mode_key"] == "TRAIN_LENGTH_MODE" and row["steps_key"] == "MAX_TRAIN_STEPS"
    assert row["modes"] == ["Epochs", "Steps"]


def test_length_info_counts_the_folder(tmp_path):
    d = _desc()
    folder = _folder(tmp_path, 7)
    info = form.length_info(d, {**form.DEFAULTS}, folder, "standard")
    assert info["items"] == 7 and info["batch"] == 1 and info["per_epoch"] == 7 and info["exact"]
    st = form.state(d, {**form.DEFAULTS}, image_folder=folder)
    assert st["length"]["per_epoch"] == 7


def test_length_info_without_a_folder():
    info = form.length_info(_desc(), {**form.DEFAULTS}, "", "standard")
    assert info["per_epoch"] == 0 and info["why"]


def test_length_steps_only_in_steps_mode():
    assert launch.length_steps({"TRAIN_LENGTH_MODE": "Epochs", "MAX_TRAIN_STEPS": "1000"}) == 0
    assert launch.length_steps({"TRAIN_LENGTH_MODE": "Steps", "MAX_TRAIN_STEPS": "1,000"}) == 1000
    assert launch.length_steps({"TRAIN_LENGTH_MODE": "Steps", "MAX_TRAIN_STEPS": ""}) == 0
    assert launch.length_steps({"TRAIN_LENGTH_MODE": "Steps", "MAX_TRAIN_STEPS": "abc"}) == 0
    assert launch.length_steps({"TRAIN_LENGTH_MODE": "Steps", "MAX_TRAIN_STEPS": "900", "FAMILY_FT": True}) == 0


def test_trainer_takes_max_train_steps():
    src = open(os.path.join(os.path.dirname(launch.__file__), "train.py"), encoding="utf-8").read()
    assert '"--max_train_steps"' in src and "max_train_steps=a.max_train_steps" in src
