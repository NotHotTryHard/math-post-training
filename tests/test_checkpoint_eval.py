from pathlib import Path

from math_post_training.checkpoint_eval import select_checkpoints


def test_select_checkpoints_filters_before_taking_every_nth(tmp_path):
    for step in (250, 500, 750, 1000, 1250):
        (tmp_path / f"checkpoint-{step}").mkdir()

    selected = select_checkpoints(tmp_path, every=2, min_step=500, max_step=1250)

    assert selected == [Path(tmp_path / "checkpoint-500"), Path(tmp_path / "checkpoint-1000")]
