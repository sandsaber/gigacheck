import click
from loguru import logger

from transformers import AutoConfig
from gigacheck.inference.src.mistral_detector import MistralDetector
from gigacheck.device_utils import detect_device


def load_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path)
    max_seq_len = getattr(config, "max_length", None) or config.max_position_embeddings

    model = MistralDetector(
        max_seq_len=max_seq_len,
        with_detr=config.with_detr,
        id2label=config.id2label,
        device=device,
    ).from_pretrained(model_path)

    return model


@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--text", type=str, required=True)
@click.option("--device", type=str, default="auto")
def main(model_path: str, text: str, device: str):
    if device == "auto":
        device = detect_device()
    model = load_model(model_path, device)
    output = model.predict(text)
    logger.info(f"[model={model_path}] {output}")


if __name__ == "__main__":
    main()
