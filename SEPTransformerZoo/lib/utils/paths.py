"""
The results-folder layout, built and parsed in one place:
    <results folder>/<model>/<phases|nophases>/dataset_<N>/t+<H>/[<seed run>/]<file>
ModelRun builds it while training; Analysis parses it back when reading results.
"""
from pathlib import Path


def results_path(results_folder: str, model_name: str, phases_tag: str, dataset_id: int, prediction_time: int) -> Path:
    return Path(results_folder) / model_name / phases_tag / f"dataset_{dataset_id}" / f"t+{prediction_time}"


def parse_results_path(results_folder, path) -> dict:
    # {root, model_type, phases, dataset, t, [model_name,] file, path}; model_name only for champion seed runs
    parts = Path(path).relative_to(results_folder).parts
    info = dict(root=str(results_folder), model_type=parts[0], phases=parts[1], dataset=parts[2], t=parts[3])
    if len(parts) == 6:
        info['model_name'] = parts[4]
    info['file'] = parts[-1]
    info['path'] = str(path)
    return info
