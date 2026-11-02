"""Create a visual review of the classifier's held-out images."""
import base64
import html
from pathlib import Path


NAMES = {0: 'Background', 1: 'Boat'}


def write_review(predictions, output):
    cards = []
    mistakes = sum(item['actual'] != item['predicted'] for item in predictions)
    # Show the errors first so they are easy to inspect.
    for item in sorted(predictions, key=lambda item: item['actual'] == item['predicted']):
        path = Path(item['path'])
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        actual, predicted = NAMES[item['actual']], NAMES[item['predicted']]
        correct = item['actual'] == item['predicted']
        status = 'Correct' if correct else 'Incorrect'
        cards.append(f'''<article class="card {'correct' if correct else 'incorrect'}">
          <div class="image"><img src="data:{mime};base64,{data}" alt="Test image: {html.escape(path.name, quote=True)}"></div>
          <div class="details"><span class="status">{status}</span>
          <p>Actual: {actual}<br>Predicted: {predicted}</p>
          <small>{html.escape(str(path))}</small></div></article>''')
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boat classifier review</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#172333;font:16px system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:32px 24px}h1{margin:0 0 8px}p{line-height:1.5}
.notice{background:#fff6df;border-left:4px solid #d59a16;padding:12px 16px;margin:22px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}
.card{background:white;border:1px solid #d6dfe6;border-radius:10px;overflow:hidden}
.card.incorrect{border:2px solid #c33636}.image{height:170px;background:#e5ebef;display:flex;align-items:center;justify-content:center;padding:12px}
img{max-width:100%;max-height:100%;object-fit:contain;image-rendering:auto}.details{padding:14px}.details p{margin:10px 0}
.status{font-weight:700;color:#23653e}.incorrect .status{color:#b52323}small{color:#526273;overflow-wrap:anywhere}
</style></head><body><main><h1>Boat classifier review</h1>
<p>TEST_COUNT test images · MISTAKES incorrect · Errors shown first</p>
<div class="notice">These images were split individually, so similar frames may appear in both sets. This doesn't measure how the model performs on a new video. The labels come from the dataset folders.</div>
<section class="grid" aria-label="Test predictions">CARDS</section></main></body></html>'''
    page = page.replace('TEST_COUNT', str(len(predictions))).replace('MISTAKES', str(mistakes)).replace('CARDS', '\n'.join(cards))
    Path(output).write_text(page, encoding='utf-8')
