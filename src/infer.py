import argparse
import os
from typing import List

import torch
import torch.nn.functional as F

from src.data.dataset import create_dataloaders
from src.data.features import IDX_TO_FINGER, FeatureBuilder
from src.data.parser import parse_fingering_file, load_pig_dataset
from src.export.musicxml import predictions_to_musicxml
from src.models.model import FingeringModel
from src.models.transformer_model import TransformerFingering
from src.utils.config import load_config
from src.utils.decoder import beam_search_decode
from src.utils.musicxml_io import parse_musicxml, write_fingerings_to_musicxml
from src.utils.midi_io import parse_midi


def forward_pass(model, main, phys, mask, phys_lambda, top_k: int):
    main_logits, phys_logits, attn = model(main, phys, mask)
    main_prob = F.softmax(main_logits, dim=-1)
    phys_prob = F.softmax(phys_logits, dim=-1)
    combined = (1 - phys_lambda) * main_prob + phys_lambda * phys_prob
    if top_k > 1:
        topk_vals, topk_idx = combined.topk(top_k, dim=-1)
        choice = torch.argmax(topk_vals, dim=-1)
        preds = torch.gather(topk_idx, dim=-1, index=choice.unsqueeze(-1)).squeeze(-1)
    else:
        preds = combined.argmax(dim=-1)
    return preds, combined, attn


def run_inference(
    config_path: str,
    checkpoint: str,
    input_path: str,
    xml_out: str,
    top_k: int,
    use_beam: bool,
    xml_template: str = None,
    midi_split: int = 60,
):
    cfg = load_config(config_path)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)

    pieces = load_pig_dataset(cfg["data"]["root"])
    builder = FeatureBuilder(
        feature_type=cfg["data"]["feature_type"],
        word2vec_dim=cfg["data"]["word2vec_dim"],
        velocity_threshold=cfg["data"]["velocity_threshold"],
    )
    if cfg["data"]["feature_type"] == "word2vec":
        builder.fit_word2vec(pieces)

    if input_path.lower().endswith((".xml", ".musicxml", ".mxl")):
        events, xml_tree = parse_musicxml(input_path)
    elif input_path.lower().endswith((".mid", ".midi")):
        xml_tree = None
        events = parse_midi(input_path, split_pitch=midi_split)
    else:
        xml_tree = None
        _, events = parse_fingering_file(input_path)
    main_feats, phys_feats, labels = builder.build(events)
    input_dim = ckpt.get("input_dim", main_feats.shape[-1])
    phys_dim = ckpt.get("phys_dim", phys_feats.shape[-1])

    arch = cfg["model"].get("arch", "cnn_bilstm")
    if arch == "transformer":
        model = TransformerFingering(
            input_dim=input_dim,
            phys_dim=phys_dim,
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["tf_layers"],
            dropout=cfg["model"]["dropout"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    else:
        model = FingeringModel(
            input_dim=input_dim,
            phys_dim=phys_dim,
            hidden_size=cfg["model"]["hidden_size"],
            cnn_channels=cfg["model"]["cnn_channels"],
            cnn_layers=cfg["model"]["cnn_layers"],
            lstm_layers=cfg["model"]["lstm_layers"],
            dropout=cfg["model"]["dropout"],
            num_classes=cfg["model"]["num_classes"],
            use_attention=cfg["model"]["attention"],
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    main_tensor = torch.tensor(main_feats, dtype=torch.float32, device=device).unsqueeze(0)
    phys_tensor = torch.tensor(phys_feats, dtype=torch.float32, device=device).unsqueeze(0)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(labels_tensor, dtype=torch.bool)

    with torch.no_grad():
        preds_idx, combined, _ = forward_pass(
            model,
            main_tensor,
            phys_tensor,
            mask,
            cfg["model"]["phys_lambda"],
            top_k=top_k,
        )
        if use_beam:
            probs = combined.squeeze(0).cpu()
            midi_list = [ev.midi for ev in events]
            channel_list = [ev.channel for ev in events]
            preds_idx = beam_search_decode(
                probs,
                midi_list,
                channel_list,
                beam_size=cfg.get("decoder", {}).get("beam_size", 5),
                alpha=cfg.get("decoder", {}).get("alpha", 0.1),
                beta=cfg.get("decoder", {}).get("beta", 0.05),
            )
        else:
            preds_idx = preds_idx.squeeze(0).cpu().tolist()
    preds_idx = preds_idx if isinstance(preds_idx, list) else preds_idx
    fingers: List[int] = [IDX_TO_FINGER.get(int(i), 0) for i in preds_idx]

    print("Predicted finger sequence (idx->finger):", fingers)
    if xml_out:
        out_path = xml_out
        if xml_template:
            tpl_events, tpl_tree = parse_musicxml(xml_template)
            if len(tpl_events) != len(fingers):
                min_len = min(len(tpl_events), len(fingers))
                print(
                    f"Warning: template notes ({len(tpl_events)}) != predicted fingers ({len(fingers)}); truncating to {min_len}"
                )
                tpl_events = tpl_events[:min_len]
                fingers = fingers[:min_len]
            write_fingerings_to_musicxml(tpl_tree, fingers, tpl_events, out_path)
        elif input_path.lower().endswith((".xml", ".musicxml", ".mxl")) and xml_tree is not None:
            write_fingerings_to_musicxml(xml_tree, fingers, events, out_path)
        else:
            out_path = predictions_to_musicxml(events, preds_idx, xml_out)
        print(f"MusicXML with fingerings written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--input", type=str, required=True, help="Path to a PIG txt, MusicXML, or MIDI file")
    parser.add_argument("--xml-out", type=str, default="outputs/musicxml/output.musicxml")
    parser.add_argument(
        "--xml-template",
        type=str,
        default=None,
        help="Optional MusicXML/MXL template to receive predicted fingerings when input is not XML",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam", action="store_true", help="Use beam search decoding with physical costs")
    parser.add_argument("--midi-split", type=int, default=60, help="Pitch threshold to split hands for single-track MIDI")
    args = parser.parse_args()
    run_inference(
        args.config,
        args.checkpoint,
        args.input,
        args.xml_out,
        args.top_k,
        args.beam,
        args.xml_template,
        args.midi_split,
    )