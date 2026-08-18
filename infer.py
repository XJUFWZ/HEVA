import argparse
import csv
import json
from pathlib import Path

from daridge.audio import FEATURE_PROTOCOL_VERSION
from daridge.audio import FrozenAudioExtractor
from daridge.inference import load_model_package
from daridge.inference import predict_feature_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict dynamic Valence/Arousal sequences for one MP3 file.",
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input MP3 file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the exported model NPZ package.",
    )
    parser.add_argument(
        "--backbone",
        required=True,
        help="Directory containing the frozen Music Flamingo weights.",
    )
    parser.add_argument(
        "--output",
        help=("Output CSV path; defaults to <audio_stem>_va.csv " "beside the input file."),
    )
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--semantic-device", default="cuda:1")
    return parser.parse_args()


def write_prediction_csv(path, prediction):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "valence", "arousal"))
        for time, valence, arousal in zip(
            prediction["time"],
            prediction["valence"],
            prediction["arousal"],
        ):
            writer.writerow(
                (
                    format(float(time), ".3f"),
                    format(float(valence), ".6f"),
                    format(float(arousal), ".6f"),
                )
            )
    return output


def validate_paths(audio_path, model_path, backbone_path, output_path):
    if audio_path.suffix.lower() != ".mp3":
        raise ValueError("--audio must point to an MP3 file")
    if not audio_path.is_file():
        raise FileNotFoundError("audio file does not exist: " + str(audio_path))
    if not model_path.is_file():
        raise FileNotFoundError("model file does not exist: " + str(model_path))
    if not backbone_path.is_dir():
        raise FileNotFoundError("Music Flamingo directory does not exist: " + str(backbone_path))
    if output_path.suffix.lower() != ".csv":
        raise ValueError("--output must point to a CSV file")
    resolved_output = output_path.resolve()
    if resolved_output in (audio_path.resolve(), model_path.resolve()):
        raise ValueError("--output must not overwrite the audio or model file")


def validate_model_protocol(package):
    package_protocol = package.metadata.get("feature_protocol_version")
    if package_protocol != FEATURE_PROTOCOL_VERSION:
        raise ValueError("model feature protocol does not match this inference code")


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    model_path = Path(args.model)
    backbone_path = Path(args.backbone)
    output_path = (
        Path(args.output) if args.output else audio_path.with_name(audio_path.stem + "_va.csv")
    )
    validate_paths(audio_path, model_path, backbone_path, output_path)
    package = load_model_package(model_path)
    validate_model_protocol(package)
    extractor = FrozenAudioExtractor(
        backbone_path=backbone_path,
        selected_heads=package.arousal.selected_heads,
        device=args.base_device,
        semantic_device=args.semantic_device,
    )
    features = extractor.extract_song(audio_path)
    prediction = predict_feature_sequence(
        package,
        features,
        track_id=audio_path.stem,
    )
    saved = write_prediction_csv(output_path, prediction)
    print(
        json.dumps(
            {
                "audio": str(audio_path),
                "model": str(model_path),
                "output": str(saved),
                "point_count": int(len(prediction["time"])),
                "time_start": float(prediction["time"][0]),
                "time_end": float(prediction["time"][-1]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
