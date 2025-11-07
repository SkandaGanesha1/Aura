import json
from pathlib import Path

from scripts import dataset_preprocess


def test_flatten_view_hierarchy(tmp_path: Path) -> None:
    tree = {
        "resource_id": "root",
        "bounds": [0, 0, 100, 100],
        "text": "",
        "children": [
            {"resource_id": "child", "bounds": [0, 0, 50, 50], "text": "Hello"},
        ],
    }
    nodes = dataset_preprocess.flatten_view_hierarchy(tree)
    assert len(nodes) == 2
    assert nodes[1]["text"] == "Hello"


def test_write_slm_training_data(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()

    (episode_dir / "view_hierarchy.json").write_text(json.dumps({"resource_id": "root"}), encoding="utf-8")
    (episode_dir / "actions.json").write_text(json.dumps({"actions": []}), encoding="utf-8")
    (episode_dir / "metadata.json").write_text(json.dumps({"instruction": "Do something"}), encoding="utf-8")

    episodes = [dataset_preprocess.parse_episode(episode_dir)]
    output_dir = tmp_path / "out"
    count = dataset_preprocess.write_slm_training_data(episodes, output_dir, min_instruction_length=1)
    assert count == 1


def test_discover_episodes_returns_unique_directories(tmp_path: Path) -> None:
    for index in range(2):
        episode_dir = tmp_path / f"episode_{index}"
        episode_dir.mkdir()

        (episode_dir / "episode.json").write_text(json.dumps({}), encoding="utf-8")
        (episode_dir / "view_hierarchy.json").write_text(
            json.dumps({"resource_id": "root"}), encoding="utf-8"
        )
        (episode_dir / "actions.json").write_text(
            json.dumps({"actions": [], "instruction": f"Instruction {index}"}), encoding="utf-8"
        )

    episodes = list(dataset_preprocess.discover_episodes(tmp_path))
    assert len(episodes) == 2
    assert {episode.instruction for episode in episodes} == {"Instruction 0", "Instruction 1"}
