import os
import time
from functools import lru_cache
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from flask import Flask, render_template_string, request

# Lazy grammar import/load to avoid HF model OOM/paging issues during server startup.
# If Phi-3 can't be loaded on this machine, we degrade gracefully.

def improve_sentence(text: str) -> str:
    try:
        from grammar import improve_sentence as _improve

        return _improve(text)
    except Exception:
        # Degrade gracefully: return raw text if Phi-3 can't load/run.
        return text

# --------------------
# Model definition
# --------------------


class Cnn(nn.Module):
    def __init__(self, input_channels, num_classes=29):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------
# App / Inference
# --------------------


app = Flask(__name__)

# Update these if you want different paths
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.getcwd(), "sign_model.pth"))

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "del", "nothing", "space",
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model():
    model = Cnn(input_channels=3, num_classes=29)
    # Using strict=False or catching FileNotFoundError for robustness if model isn't present
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found. Running with untrained weights for demonstration.")
    model = model.to(device)
    model.eval()
    return model


def predict_letter(image: Image.Image) -> str:
    model = load_model()

    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        pred_idx = outputs.argmax(dim=1).item()

    return CLASSES[pred_idx]


def apply_sequence_rules(current_text: str, letter: str) -> str:
    if letter == "space":
        return current_text + " "
    if letter == "del":
        return current_text[:-1]
    if letter == "nothing":
        return current_text
    return current_text + letter

# Modern Tailwind CSS + Jinja2 Template
PAGE = """
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>ASL to Text Translator</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    /* Custom scrollbar for pre tags */
    pre::-webkit-scrollbar { height: 8px; }
    pre::-webkit-scrollbar-track { background: transparent; }
    pre::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 4px; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
  
  <div class="max-w-3xl w-full space-y-8">
    
    <!-- Header -->
    <div class="text-center">
      <div class="inline-flex items-center justify-center p-3 bg-indigo-100 rounded-full mb-4">
        <svg class="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11"></path>
        </svg>
      </div>
      <h2 class="text-3xl font-extrabold text-slate-900 tracking-tight">ASL Sign Language → Text</h2>
      <p class="mt-2 text-slate-500">Upload an image of an ASL sign. The model predicts the letter and a Phi-3 model refines the grammar.</p>
    </div>

    <!-- Upload Card -->
    <div class="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 transition-all hover:shadow-md">
      <form method='POST' action='/predict' enctype='multipart/form-data' class="space-y-6">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-2">Upload Images (sequence)</label>
          <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-300 border-dashed rounded-xl hover:border-indigo-400 transition-colors group">
            <div class="space-y-1 text-center">
              <svg class="mx-auto h-12 w-12 text-slate-400 group-hover:text-indigo-500 transition-colors" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
              <div class="flex text-sm text-slate-600 justify-center mt-4">
                <label for="file-upload" class="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                  <span>Choose files</span>
                  <input id="file-upload" name="files" type="file" class="sr-only" accept="image/*" multiple required>
                </label>
                <p class="pl-1">(Select in order by filename)</p>
              </div>
              <p class="text-xs text-slate-500">PNG, JPG, GIF up to 10MB each</p>
            </div>
          </div>
        </div>
        
        <div class="flex items-center justify-between pt-2">
          <span class="text-sm text-slate-500" id="file-name-display">No files chosen</span>
          <button type='submit' class="inline-flex justify-center py-2.5 px-6 border border-transparent shadow-sm text-sm font-medium rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors">
            Analyze Sequence
          </button>
        </div>
      </form>
    </div>

    <!-- Script to display selected file name -->
    <script>
      document.getElementById('file-upload').addEventListener('change', function(e) {
        var fileName = e.target.files[0] ? e.target.files[0].name : "No file chosen";
        document.getElementById('file-name-display').textContent = fileName;
      });
    </script>

    <!-- Results Section -->
    {% if result %}
    <div class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden animate-fade-in-up">
      <div class="px-6 py-5 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
        <h3 class="text-lg leading-6 font-medium text-slate-900">Analysis Results</h3>
        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          Success ({{ result.inference_ms }}ms)
        </span>
      </div>
      <div class="p-6 space-y-6">
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="col-span-1 flex flex-col items-center justify-center p-6 bg-indigo-50 rounded-xl border border-indigo-100">
            <span class="text-sm font-medium text-indigo-800 mb-1">Predicted Class</span>
            <span class="text-5xl font-black text-indigo-600">{{ result.predicted_letter }}</span>
          </div>
          
          <div class="col-span-2 space-y-4">
            <div>
              <h4 class="text-sm font-medium text-slate-500 mb-1">Raw Text Output</h4>
              <div class="w-full bg-slate-100 p-3 rounded-lg text-slate-800 font-mono text-sm border border-slate-200">
                {{ result.raw_text }}
              </div>
            </div>
            
            <div>
              <h4 class="text-sm font-medium text-slate-500 mb-1">Improved Sentence (Phi-3)</h4>
              <pre class="w-full bg-slate-800 p-4 rounded-lg text-emerald-400 font-mono text-sm shadow-inner whitespace-pre-wrap">{{ result.final_sentence }}</pre>
            </div>
          </div>
        </div>

      </div>
    </div>
    {% endif %}

    <!-- Error Section -->
    {% if error %}
    <div class="bg-red-50 border-l-4 border-red-500 p-6 rounded-r-xl shadow-sm animate-fade-in-up">
      <div class="flex">
        <div class="flex-shrink-0">
          <svg class="h-6 w-6 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
          </svg>
        </div>
        <div class="ml-3">
          <h3 class="text-sm font-medium text-red-800">Processing Error</h3>
          <div class="mt-2 text-sm text-red-700 font-mono bg-red-100/50 p-3 rounded-md">
            {{ error }}
          </div>
        </div>
      </div>
    </div>
    {% endif %}

  </div>
  
  <style>
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in-up { animation: fadeInUp 0.4s ease-out forwards; }
  </style>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict_route():
    start = time.time()
    try:
        # Sequence mode (multiple uploaded images)
        if "files" not in request.files:
            return render_template_string(PAGE, result=None, error="Missing files field."), 400

        files = request.files.getlist("files")
        files = [f for f in files if f and f.filename]
        if not files:
            return render_template_string(PAGE, result=None, error="No images selected."), 400

        # Predict in filename order for stable sequence
        files_sorted = sorted(files, key=lambda x: x.filename.lower())

        raw_text = ""
        last_letter = ""
        for f in files_sorted:
            img = Image.open(f.stream)
            letter = predict_letter(img)
            last_letter = letter

            # replicate your prediction.py sequencing behavior
            if letter == "space":
                raw_text += " "
            elif letter == "del":
                raw_text = raw_text[:-1]
            elif letter == "nothing":
                continue
            else:
                raw_text += letter

        final_sentence = improve_sentence(raw_text.strip() if raw_text else last_letter)



        inference_ms = int((time.time() - start) * 1000)

        result = {
            "predicted_letter": letter,
            "raw_text": raw_text,
            "final_sentence": final_sentence,
            "inference_ms": inference_ms,
        }
        return render_template_string(PAGE, result=result, error=None)

    except Exception as e:
        inference_ms = int((time.time() - start) * 1000)
        return (
            render_template_string(PAGE, result=None, error=f"{type(e).__name__}: {e}\n\n(inference_ms={inference_ms})"),
            500,
        )


if __name__ == "__main__":
    # For production, use gunicorn/uwsgi; debug=True is for dev only.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
