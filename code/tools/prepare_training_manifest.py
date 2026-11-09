#!/usr/bin/env python3
"""Freeze local Stage 1/2 inputs into audited, directly loadable JSONL files.

Paths in the input config are resolved relative to the config file. Each source
accepts sample_count, sampling_rate, or selection_indices (one integer per line).
Without a selection_indices file, sampling is a NEW deterministic local
selection; the paper's historical VQA subset identities are not reconstructed
by choosing the same count. No RGB, video, instruction, or action is generated
or relabelled by this tool.

Examples (run from code/):
  python tools/prepare_training_manifest.py --config config/training_stage1.example.json --output-dir ../training/stage1
  python tools/prepare_training_manifest.py --verify ../training/stage1/manifest.json
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path


FORMAT = "guidednav_navigation_training_v1"
ACTION_TEXT = {0: "STOP", 1: "\u2191", 2: "\u2190", 3: "\u2192"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Each local path must be a nonempty string")
    return (base / os.path.expandvars(os.path.expanduser(value))).resolve()


def _annotation_files(base, patterns):
    patterns = [patterns] if isinstance(patterns, str) else patterns
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("annotation_path must be a path or a nonempty list of globs")
    resolved = set()
    for pattern in patterns:
        absolute = str(_path(base, pattern))
        matches = sorted(glob.glob(absolute, recursive=True))
        if not matches:
            raise FileNotFoundError("No annotation files matched: " + absolute)
        resolved.update(Path(match).resolve() for match in matches)
    files = sorted(resolved, key=str)
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("Expected a local .json or .jsonl annotation file: " + str(path))
    return files


def iter_records(path):
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON at {}:{}".format(path, line_number)) from exc
    else:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("JSON annotations must contain a list: " + str(path))
        yield from records


def _media_values(record, key):
    values = record.get(key, [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(key + " must be a local path or a list of local paths")
    return values


def _conversations(record):
    turns = record.get("conversations")
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("conversations must contain at least one human/gpt pair")
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("from") != ("human" if index % 2 == 0 else "gpt"):
            raise ValueError("conversations must alternate human and gpt turns")
        if not isinstance(turn.get("value"), str) or not turn["value"].strip():
            raise ValueError("Each conversation value must be nonempty text")
    if len(turns) % 2:
        raise ValueError("A training conversation must end with a gpt answer")
    return turns


def validate_record(record, source, stage):
    """Validate input contracts; return media paths without altering a record."""
    if not isinstance(record, dict):
        raise ValueError("Each row must be an individual object; unpack grouped/packed VQA rows first")
    turns = _conversations(record)
    if source["type"] == "vqa":
        images, videos = _media_values(record, "image"), _media_values(record, "video")
        if not images and not videos:
            raise ValueError("VQA rows need image or video media")
        human_text = " ".join(turn["value"] for turn in turns if turn["from"] == "human")
        image_tokens, video_tokens = human_text.count("<image>"), human_text.count("<video>")
        # The inherited VQA loader rewrites these two upstream video datasets'
        # historical <image> placeholders into <video> before tokenization.
        legacy_video = (not images and videos and source["name"] in {"vsi_590k", "llava_video_178k"})
        if legacy_video:
            if image_tokens + video_tokens != len(videos):
                raise ValueError("VQA video placeholders do not match the video field")
        elif image_tokens != len(images) or video_tokens != len(videos):
            raise ValueError("VQA image/video placeholders do not match media fields")
        return images + videos

    images = record.get("images")
    if not isinstance(images, list) or not images or len(images) > 8 or any(not isinstance(item, str) or not item for item in images):
        raise ValueError("VLN images must contain 1 to 8 history/current frame paths")
    if type(record.get("n_memory")) is not int or record["n_memory"] != len(images) - 1:
        raise ValueError("VLN n_memory must equal len(images)-1")
    if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
        raise ValueError("VLN instruction must be nonempty text")
    if type(record.get("c")) is not int or record["c"] < 0:
        raise ValueError("VLN c must be a zero-based nonnegative observation index")
    if not Path(images[-1]).stem.isdigit() or int(Path(images[-1]).stem) != record["c"] + 1:
        raise ValueError("Current RGB filename must use c+1 indexing (001.jpg for c=0)")
    ids = record.get("action_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 4 or any(type(item) is not int or item not in ACTION_TEXT for item in ids):
        raise ValueError("action_ids must contain 1 to 4 discrete action IDs in [0,3]")
    expected = "".join(ACTION_TEXT[item] for item in ids)
    if record.get("action_text") != expected or len(turns) != 2 or turns[-1]["value"] != expected:
        raise ValueError("Stage 1/2 navigation must supervise action-only glyphs matching action_ids")
    if record.get("subtask") or record.get("phase_mode"):
        raise ValueError("Subtask/phase targets do not belong in Stage 1/2")
    human = turns[0]["value"]
    if human.count("<image>") != 1 or human.count("<memory>") != int(record["n_memory"] > 0):
        raise ValueError("Navigation conversations need one current image and an optional memory placeholder")
    if stage == 2 and record.get("action_source") != "expert_chunks":
        raise ValueError("Stage 2 DAgger labels must come from expert_chunks, never executed policy actions")
    if stage == 1 and record.get("action_source", "actions") != "actions":
        raise ValueError("Stage 1 expects expert trajectories built with action_source=actions")
    return images


def _selection(source, count, base, seed):
    choices = [key for key in ("sample_count", "sampling_rate", "selection_indices") if key in source]
    if len(choices) > 1:
        raise ValueError("Use only one of sample_count, sampling_rate, selection_indices per source")
    if "selection_indices" in source:
        indices_path = _path(base, source["selection_indices"])
        values = []
        with open(indices_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if type(value) is not int:
                    raise ValueError("selection_indices needs one integer source-row index per line")
                values.append(value)
        if len(values) != len(set(values)) or any(value < 0 or value >= count for value in values):
            raise ValueError("selection_indices contains a duplicate or out-of-range index")
        indices = sorted(values)
        origin = "supplied_source_indices"
    else:
        rate = source.get("sampling_rate", 1.0)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("sampling_rate must be in (0,1]")
        selected_count = source.get("sample_count", int(count * rate))
        if type(selected_count) is not int or not 1 <= selected_count <= count:
            raise ValueError("sample_count must be an integer between 1 and the source row count")
        indices = range(count) if selected_count == count else sorted(random.Random(seed).sample(range(count), selected_count))
        origin = "all_source_rows" if selected_count == count else "new_local_deterministic_selection"
    if not len(indices):
        raise ValueError("A source selection must contain at least one row")
    return indices, origin


def prepare(config_path, output_dir, check_media="all"):
    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    stage, seed = config.get("stage"), config.get("seed", 42)
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("This tool prepares Stage 1 or Stage 2 inputs")
    if type(seed) is not int or seed != 42:
        raise ValueError("The paper training seed is 42")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a nonempty list")
    names = [source.get("name") for source in sources]
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names) or len(names) != len(set(names)):
        raise ValueError("Source names must be unique filename-safe names")
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; existing data will not be overwritten: " + str(output_dir))
    output_dir.mkdir(parents=True)
    prepared = []
    for source in sources:
        name, kind = source["name"], source.get("type")
        if kind not in {"vqa", "vln_unified"}:
            raise ValueError("Unknown source type: " + str(kind))
        expected_role = "visual_qa" if kind == "vqa" else ("expert_navigation" if stage == 1 else "dagger_navigation")
        if source.get("role", expected_role) != expected_role:
            raise ValueError("Source {} has an incompatible role for Stage {}".format(name, stage))
        if kind == "vqa":
            files = _annotation_files(config_path.parent, source["annotation_path"])
            media_root = _path(config_path.parent, source["data_path"])
        else:
            files = [_path(config_path.parent, source["samples_path"])]
            if files[0].suffix.lower() != ".jsonl":
                raise ValueError("VLN samples_path must be a JSONL file")
            media_root = _path(config_path.parent, source["video_folder"])
        if not media_root.is_dir():
            raise NotADirectoryError("Missing media root: " + str(media_root))
        file_receipts = []
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = sum(1 for _ in iter_records(path))
            file_receipts.append({"path": str(path), "rows": rows, "sha256": sha256_file(path)})
        count = sum(item["rows"] for item in file_receipts)
        indices, origin = _selection(source, count, config_path.parent, seed)
        selection = iter(indices)
        wanted = next(selection, None)
        output_file, indices_file = output_dir / (name + ".jsonl"), output_dir / (name + ".source_indices.jsonl")
        media_checked, selected_count = 0, 0
        with open(output_file, "x", encoding="utf-8", newline="\n") as out, open(indices_file, "x", encoding="utf-8", newline="\n") as index_out:
            source_index = 0
            for path in files:
                for record in iter_records(path):
                    if source_index == wanted:
                        try:
                            media = validate_record(record, source, stage)
                            if check_media == "all" or (check_media == "sample" and selected_count < 10):
                                for relative in media:
                                    asset = _path(media_root, relative)
                                    if not asset.exists():
                                        raise FileNotFoundError("Missing RGB/video asset: " + str(asset))
                                    media_checked += 1
                        except (ValueError, FileNotFoundError) as exc:
                            raise ValueError("{} source row {}: {}".format(name, source_index, exc)) from exc
                        out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        index_out.write(str(source_index) + "\n")
                        selected_count += 1
                        wanted = next(selection, None)
                    source_index += 1
        if selected_count != len(indices):
            raise ValueError("Source row count changed during preparation: " + name)
        # Detect edits made between counting and export. Changed inputs must
        # never acquire a receipt for their previous version.
        if any(sha256_file(item["path"]) != item["sha256"] for item in file_receipts):
            raise ValueError("Input annotation file changed during preparation: " + name)
        item = {
            "name": name, "type": kind, "role": expected_role,
            "sampling_rate": 1.0, "selected_rows": selected_count,
            "sha256": sha256_file(output_file), "source_files": file_receipts,
            "selection": {
                "identity": origin, "seed": seed,
                "algorithm": "sorted CPython random.Random(42).sample source-row indices" if origin == "new_local_deterministic_selection" else origin,
                "source_rows": count, "selected_rows": selected_count,
                "indices_path": str(indices_file), "indices_sha256": sha256_file(indices_file),
            },
            "media_check": {"mode": check_media, "checked_references": media_checked},
        }
        if kind == "vqa":
            item.update(annotation_path=str(output_file), data_path=str(media_root))
        else:
            item.update(samples_path=str(output_file), video_folder=str(media_root))
        prepared.append(item)
        print("{}: {} / {} source rows".format(name, selected_count, count), flush=True)
    manifest = {
        "format": FORMAT, "prepared": True, "stage": stage, "seed": seed,
        "selected_rows": sum(item["selected_rows"] for item in prepared),
        "config_sha256": sha256_file(config_path),
        "selection_note": "Locally frozen inputs. Matching published sample counts alone does not establish identity with the original training subsets.",
        "sources": prepared,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path


def verify_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FORMAT or manifest.get("prepared") is not True or manifest.get("stage") not in (1, 2) or manifest.get("seed") != 42:
        raise ValueError("Not a prepared Stage 1/2 training manifest")
    total = 0
    for source in manifest["sources"]:
        if source.get("sampling_rate") != 1.0:
            raise ValueError("Prepared inputs must not be sampled again by the loader")
        records = Path(source["annotation_path"] if source["type"] == "vqa" else source["samples_path"])
        if sha256_file(records) != source["sha256"]:
            raise ValueError("Prepared source hash mismatch: " + source["name"])
        selection = source["selection"]
        if sha256_file(selection["indices_path"]) != selection["indices_sha256"]:
            raise ValueError("Selection index hash mismatch: " + source["name"])
        count = sum(1 for _ in iter_records(records))
        with open(selection["indices_path"], "r", encoding="utf-8") as handle:
            indices = [json.loads(line) for line in handle if line.strip()]
        if count != source["selected_rows"] or count != len(indices) or count != selection["selected_rows"]:
            raise ValueError("Prepared source/index row count mismatch: " + source["name"])
        if any(type(index) is not int or index < 0 or index >= selection["source_rows"] for index in indices) or indices != sorted(set(indices)):
            raise ValueError("Invalid frozen source indices: " + source["name"])
        for record in iter_records(records):
            validate_record(record, source, manifest["stage"])
        total += count
    if total != manifest["selected_rows"]:
        raise ValueError("Manifest total row count mismatch")
    return {"stage": manifest["stage"], "sources": len(manifest["sources"]), "selected_rows": total, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-media", choices=("all", "sample", "none"), default="all", help="all checks every selected media reference; sample checks the first 10 selected rows per source")
    args = parser.parse_args()
    if args.verify:
        if args.output_dir:
            parser.error("--output-dir is used with --config")
        print(json.dumps(verify_manifest(args.verify), indent=2))
    else:
        if not args.output_dir:
            parser.error("--config requires --output-dir")
        print(prepare(args.config, args.output_dir, args.check_media))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
