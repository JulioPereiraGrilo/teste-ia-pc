# ======== app.py (1 chamada ao Gemini + sprite direita-apenas, SEM DEMO) ========
import os
import json
import uuid
import re
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
import google.generativeai as genai

# ========== CONFIG ==========
load_dotenv()
API_KEY = os.getenv("API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY/GOOGLE_API_KEY não encontrada no .env. Defina sua chave antes de iniciar o servidor.")
genai.configure(api_key=API_KEY)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Modelo: use "gemini-2.5-pro" para máxima precisão (recomendado)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Ordem das avarias (10 linhas, topo -> base)
LINES = [
    "aerosaculite","celulite","contusao","hematomas","hepatite",
    "micoplasmose","pericardite","peritonite","salmonelose","tuberculose"
]

# ======== PARÂMETROS DE SPRITE (só Pillow) ========
TOP_RATIO = 0.08          # corta 8% do topo
BOTTOM_RATIO = 0.08       # corta 8% da base
CENTER_SHIFT_PCT = 0.035  # puxa o "centro" 3.5% para ESQUERDA
RIGHT_FRACTION = 0.55     # recorta ~55% finais (mais que metade)
RIGHT_ZOOM = 1.6          # zoom nos recortes da direita
SPRITE_W = 1000           # largura de cada faixa no sprite
SPRITE_H = 130            # altura de cada faixa no sprite
LABEL_W = 120             # coluna de rótulo (1..10)

# ========== PROMPT ÚNICO (original + sprite direita-apenas) ==========
ANALYZER_PROMPT = """
Você é uma IA especializada em leitura de ábacos físicos usados em inspeção industrial.

Você receberá duas imagens:
(1) Ábaco original (contexto).
(2) SPRITE com 10 recortes, de cima para baixo (linhas 1..10), cada recorte contém APENAS o LADO DIREITO do observador,
   já ampliado e nitidado para facilitar a contagem.

Regras:
- Leia SEMPRE do ponto de vista do observador: a contagem válida é a das bolinhas à DIREITA.
- Leia as linhas de CIMA para BAIXO (topo → base) e mapeie assim:
  1: aerosaculite
  2: celulite
  3: contusao
  4: hematomas
  5: hepatite
  6: micoplasmose
  7: pericardite
  8: peritonite
  9: salmonelose
  10: tuberculose

Procedimento por linha:
1) Conte DIRETA e explicitamente no sprite quantas bolinhas estão à DIREITA.
2) (Checagem) Opcional: usando o original, conte à ESQUERDA e verifique se direita == 10 - esquerda.
3) Se houver dúvida, recalcule até coincidir. Use apenas inteiros 0..10.

Retorne SOMENTE JSON (sem texto extra), exatamente neste formato:
{
  "linhas": [
    {"linha": 1, "avaria": "aerosaculite", "direita": <int>},
    {"linha": 2, "avaria": "celulite", "direita": <int>},
    {"linha": 3, "avaria": "contusao", "direita": <int>},
    {"linha": 4, "avaria": "hematomas", "direita": <int>},
    {"linha": 5, "avaria": "hepatite", "direita": <int>},
    {"linha": 6, "avaria": "micoplasmose", "direita": <int>},
    {"linha": 7, "avaria": "pericardite", "direita": <int>},
    {"linha": 8, "avaria": "peritonite", "direita": <int>},
    {"linha": 9, "avaria": "salmonelose", "direita": <int>},
    {"linha": 10, "avaria": "tuberculose", "direita": <int>}
  ],
  "contagem": {
    "aerosaculite": <int>,
    "celulite": <int>,
    "contusao": <int>,
    "hematomas": <int>,
    "hepatite": <int>,
    "micoplasmose": <int>,
    "pericardite": <int>,
    "peritonite": <int>,
    "salmonelose": <int>,
    "tuberculose": <int>
  }
}
"""

# ========== UTILS ==========
def save_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return path, filename

def parse_json_safely(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

def normalize_counts_map(raw: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for k in LINES:
        v = raw.get(k, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        out[k] = max(0, min(10, v))
    return out

def upload_local_image(path: str):
    return genai.upload_file(path)

def gemini_generate_json(model_name: str, parts: list, timeout_s: int = 50) -> Dict[str, Any]:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(parts, request_options={"timeout": timeout_s})
    text = (resp.text or "").strip()
    data = parse_json_safely(text)
    return data if isinstance(data, dict) else {"_raw": text}

# ========== SPRITE AUXILIAR (Pillow) ==========
def _enhance_right(im: Image.Image) -> Image.Image:
    """
    Pipeline leve para melhorar legibilidade:
    - autocontrast global
    - equalize (histograma)
    - mediana leve (ruído)
    - contrast/sharpness
    """
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageOps.equalize(im)
    im = im.filter(ImageFilter.MedianFilter(size=3))
    im = ImageEnhance.Contrast(im).enhance(1.15)
    im = ImageEnhance.Sharpness(im).enhance(1.35)
    return im

def build_right_only_sprite(im: Image.Image) -> Tuple[str, List[str]]:
    """
    Gera sprite 10x (topo→base), cada recorte = lado DIREITO (55% finais), com
    leve deslocamento do centro (CENTER_SHIFT_PCT), zoom e nitidez.
    """
    w, h = im.size
    top = int(h * TOP_RATIO)
    bot = int(h * (1 - BOTTOM_RATIO))
    if bot <= top + 10:
        top, bot = 0, h

    usable_h = bot - top
    band_h = max(1, usable_h // 10)

    sprite = Image.new("RGB", (LABEL_W + SPRITE_W, SPRITE_H * 10), (255, 255, 255))
    draw = ImageDraw.Draw(sprite)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()

    right_crops: List[str] = []
    for i in range(10):
        y1 = top + i * band_h
        y2 = bot if i == 9 else y1 + band_h
        band = im.crop((0, y1, w, y2))

        # posição de "centro" deslocado um pouco à esquerda
        center_x = int(w * (0.5 - CENTER_SHIFT_PCT))
        center_x = max(0, min(w-1, center_x))

        # largura do lado direito ampliada (RIGHT_FRACTION do total)
        right_x0 = max(0, int(w - int(w * RIGHT_FRACTION)))
        right_x0 = min(right_x0, center_x)  # garante que inclui/ultrapassa o "centro"
        right = band.crop((right_x0, 0, w, band.height))

        # zoom + melhorias
        if RIGHT_ZOOM != 1.0:
            right = right.resize((int(right.width * RIGHT_ZOOM), int(right.height * RIGHT_ZOOM)), Image.LANCZOS)
        right = _enhance_right(right)
        right = right.resize((SPRITE_W, SPRITE_H), Image.LANCZOS)

        # salva faixa individual (útil p/ depuração)
        crop_path = os.path.join(UPLOAD_FOLDER, f"right_{i+1:02d}_{uuid.uuid4().hex[:6]}.jpg")
        right.save(crop_path, quality=92)
        right_crops.append(crop_path)

        # desenha no sprite
        y_off = i * SPRITE_H
        sprite.paste(right, (LABEL_W, y_off))
        draw.text((10, y_off + 8), f"{i+1}", fill=(0,0,0), font=font)

    sprite_path = os.path.join(UPLOAD_FOLDER, f"sprite_right_{uuid.uuid4().hex[:8]}.jpg")
    sprite.save(sprite_path, quality=92)
    return sprite_path, right_crops

# ========== API ==========
@app.route("/api/contar", methods=["POST"])
def api_contar():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Nenhuma imagem enviada"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "Nenhuma imagem selecionada"}), 400

    try:
        # salva original
        path, fname = save_uploaded_file(f)
        im = Image.open(path).convert("RGB")

        # gera sprite direita-apenas
        sprite_path, crop_paths = build_right_only_sprite(im)

        # 1 chamada Gemini com [prompt, original, sprite]
        up_orig = upload_local_image(path)
        up_sprite = upload_local_image(sprite_path)

        result = gemini_generate_json(
            MODEL,
            [ANALYZER_PROMPT, up_orig, up_sprite],
            timeout_s=50
        )

        contagem = result.get("contagem", {})
        # normaliza (0..10 e garante todas as chaves)
        counts = normalize_counts_map(contagem)

        # limpeza
        for up in [up_orig, up_sprite]:
            try:
                genai.delete_file(up.name)
            except Exception:
                pass
        for p in [path, sprite_path, *crop_paths]:
            try:
                os.remove(p)
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "counts": counts,
            "model": MODEL,
            "raw_debug": result  # deixe durante a calibração; remova em produção se quiser
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ========== HEALTH ==========
@app.route("/api/health", methods=["GET"])
def health():
    try:
        models = [m.name for m in genai.list_models()]
    except Exception as e:
        models = [str(e)]
    return jsonify({
        "status": "online",
        "available_models": models,
        "model": MODEL
    })

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)