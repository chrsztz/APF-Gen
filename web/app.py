import os
import tempfile
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from src.data.features import FeatureBuilder
from src.data.parser import load_pig_dataset
from src.export.musicxml import predictions_to_musicxml
from src.infer import forward_pass
from src.models.model import FingeringModel
from src.models.transformer_model import TransformerFingering
from src.utils.config import load_config
from src.data.parser import parse_fingering_file
from src.utils.decoder import beam_search_decode
from src.utils.midi_io import parse_midi_with_meta

APP_ROOT = Path(__file__).resolve().parent
HTML_PATH = APP_ROOT / "interactive_demo.html"

CONFIG_PATH = os.getenv("PIG_CONFIG", "configs/default.yaml")
CHECKPOINT_PATH = os.getenv("PIG_CHECKPOINT", "outputs_bi/checkpoints/best.pt")

app = FastAPI(title="PIG Fingering Demo")


def _init_engine():
    cfg = load_config(CONFIG_PATH)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    builder = FeatureBuilder(
        feature_type=cfg["data"]["feature_type"],
        word2vec_dim=cfg["data"]["word2vec_dim"],
        velocity_threshold=cfg["data"]["velocity_threshold"],
    )
    if cfg["data"]["feature_type"] == "word2vec":
        pieces = load_pig_dataset(cfg["data"]["root"])
        builder.fit_word2vec(pieces)

    input_dim = ckpt.get("input_dim")
    phys_dim = ckpt.get("phys_dim")
    if input_dim is None or phys_dim is None:
        # Build a tiny sample to infer dims if ckpt doesn't include them
        sample_pieces = load_pig_dataset(cfg["data"]["root"])
        first_key = next(iter(sample_pieces))
        main_feats, phys_feats, _ = builder.build(sample_pieces[first_key])
        input_dim = input_dim or main_feats.shape[-1]
        phys_dim = phys_dim or phys_feats.shape[-1]

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
    return cfg, device, model, builder


CFG, DEVICE, MODEL, BUILDER = _init_engine()


def _predict_finger_indices(events, top_k: int, use_beam: bool) -> List[int]:
    main_feats, phys_feats, labels = BUILDER.build(events)
    main_tensor = torch.tensor(main_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    phys_tensor = torch.tensor(phys_feats, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE).unsqueeze(0)
    mask = torch.ones_like(labels_tensor, dtype=torch.bool)

    with torch.no_grad():
        preds_idx, combined, _ = forward_pass(
            MODEL,
            main_tensor,
            phys_tensor,
            mask,
            CFG["model"]["phys_lambda"],
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
                beam_size=CFG.get("decoder", {}).get("beam_size", 5),
                alpha=CFG.get("decoder", {}).get("alpha", 0.1),
                beta=CFG.get("decoder", {}).get("beta", 0.05),
            )
        else:
            preds_idx = preds_idx.squeeze(0).cpu().tolist()

    return preds_idx if isinstance(preds_idx, list) else list(preds_idx)


def _midi_bytes_to_musicxml(midi_bytes: bytes, midi_split: int, top_k: int, use_beam: bool) -> str:
    tmp_midi_path = None
    tmp_xml_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
            tmp.write(midi_bytes)
            tmp_midi_path = tmp.name

        events, midi_meta = parse_midi_with_meta(tmp_midi_path, split_pitch=midi_split)
        if not events:
            raise HTTPException(status_code=400, detail="Empty MIDI file or no note events.")

        preds_idx = _predict_finger_indices(events, top_k=top_k, use_beam=use_beam)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as tmp_xml:
            tmp_xml_path = tmp_xml.name
        predictions_to_musicxml(
            events,
            preds_idx,
            tmp_xml_path,
            title="PIG Fingering Demo",
            midi_meta=midi_meta,
        )
        return Path(tmp_xml_path).read_text(encoding="utf-8")
    finally:
        if tmp_midi_path and os.path.exists(tmp_midi_path):
            os.remove(tmp_midi_path)
        if tmp_xml_path and os.path.exists(tmp_xml_path):
            os.remove(tmp_xml_path)


def _pig_txt_to_musicxml(txt_bytes: bytes, top_k: int, use_beam: bool) -> str:
    tmp_txt_path = None
    tmp_xml_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
            tmp.write(txt_bytes)
            tmp_txt_path = tmp.name

        _, events = parse_fingering_file(tmp_txt_path)
        if not events:
            raise HTTPException(status_code=400, detail="No note events found in PIG txt file.")

        preds_idx = _predict_finger_indices(events, top_k=top_k, use_beam=use_beam)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as tmp_xml:
            tmp_xml_path = tmp_xml.name
        predictions_to_musicxml(events, preds_idx, tmp_xml_path, title="PIG Fingering Demo")
        return Path(tmp_xml_path).read_text(encoding="utf-8")
    finally:
        if tmp_txt_path and os.path.exists(tmp_txt_path):
            os.remove(tmp_txt_path)
        if tmp_xml_path and os.path.exists(tmp_xml_path):
            os.remove(tmp_xml_path)


@app.post("/api/infer_txt")
async def infer_txt(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    beam: bool = Form(False),
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a PIG fingering .txt file.")
    txt_bytes = await file.read()
    if not txt_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    xml_text = _pig_txt_to_musicxml(txt_bytes, top_k=top_k, use_beam=beam)
    return Response(content=xml_text, media_type="application/xml")


@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists():
        return HTMLResponse("<h3>interactive_demo.html not found.</h3>")
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    beam: bool = Form(False),
    midi_split: int = Form(60),
):
    if not file.filename.lower().endswith((".mid", ".midi")):
        raise HTTPException(status_code=400, detail="Please upload a MIDI file (.mid/.midi).")

    midi_bytes = await file.read()
    if not midi_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    xml_text = _midi_bytes_to_musicxml(
        midi_bytes=midi_bytes,
        midi_split=midi_split,
        top_k=top_k,
        use_beam=beam,
    )
    return Response(content=xml_text, media_type="application/xml")
