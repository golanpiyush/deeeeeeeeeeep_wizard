# DepthWizard — Single-View Height Estimation & 3D Flythrough

**SIH26175** — Upload a single 2D image, get a depth map and an interactive
3D scene you can rotate/fly through in the browser.

## How it works (plain-language overview)

1. You upload one flat 2D photo (a normal photo, or later, a lunar/satellite image).
2. A pretrained AI model called **Depth Anything V2** looks at the photo and
   guesses "how far away is every pixel from the camera" — this produces a
   **depth map** (a grayscale image where brighter = closer, darker = farther,
   or vice versa depending on convention).
3. We combine the original photo's colors with the depth map to build a
   **3D point cloud** — basically millions of tiny colored dots placed in 3D
   space, each pushed backward/forward based on how "deep" that pixel is.
4. That point cloud is sent to the browser and rendered with **Three.js**,
   a JavaScript 3D graphics library. The judge can then click-drag to rotate,
   zoom, and "fly" around the scene live, on stage, with no video pre-recording.

## Why this architecture (not Unity)

We considered Unity, but skipped it for v1 because:
- It requires a second tech stack (C#, Unity Editor, asset export pipeline)
  that most ML-focused teams don't already know.
- Three.js runs directly in any browser — zero install for judges, and your
  laptop just needs to run a local web server.
- It's just as visually impressive for a point-cloud flythrough, and it's
  dramatically faster to build and iterate on.
- It scales to the cloud later with no frontend changes (see "Scaling" below).

## Project structure

```
depthwizard/
├── README.md                  <- you are here
├── backend/
│   ├── requirements.txt       <- Python dependencies
│   ├── depth_model.py         <- loads Depth Anything V2, runs inference
│   ├── pointcloud.py          <- converts (image + depth map) -> 3D point cloud
│   ├── server.py              <- FastAPI server: upload image -> get point cloud JSON
│   └── generate_sample_depth.py  <- CPU-only fallback depth generator (for testing
│                                    the pipeline without a GPU or model download)
├── frontend/
│   ├── index.html             <- upload UI + Three.js 3D viewer (single file, no build step)
├── sample_images/             <- put test photos / lunar images here
└── outputs/                   <- generated depth maps & point clouds land here
```

## Setup (on your laptop — RTX 4060, 24GB RAM)

```bash
cd depthwizard/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# First run will auto-download the Depth Anything V2 model weights (~100-400MB
# depending on size chosen) from Hugging Face. Needs internet once.
python server.py
```

Then open `frontend/index.html` directly in your browser (or serve it with
`python -m http.server` from the `frontend/` folder), upload an image, and
watch the 3D scene render.

## Model size guide (for your 4060 6GB)

Depth Anything V2 ships in three sizes. On a 6GB laptop GPU:

| Model  | VRAM needed | Speed        | Quality   | Recommended for      |
|--------|-------------|--------------|-----------|-----------------------|
| Small  | ~1-2 GB     | Fastest      | Good      | Live demo, fast iteration |
| Base   | ~2-4 GB     | Medium       | Better    | Final polished demo  |
| Large  | ~6-8 GB     | Slow, tight  | Best      | Risky on 6GB — may OOM |

**Recommendation: start with Small, switch to Base once the pipeline works
end-to-end.** Avoid Large on a 6GB card — too close to the VRAM ceiling
alongside the rest of your OS/browser overhead.

## Scaling to the cloud later

The backend is a plain FastAPI service. To scale later:
- Swap local inference in `depth_model.py` for a hosted GPU endpoint
  (e.g. Modal, RunPod, a rented cloud GPU box, or a HF Inference Endpoint).
- The `server.py` API contract (`POST /predict` -> point cloud JSON) stays
  identical, so the frontend never needs to change.
- For handling many judges/users at once later, put a queue (e.g. Redis +
  a worker) in front of the model — not needed for a hackathon demo.

## Roadmap after this v1

1. Get generic photos working end-to-end (this v1).
2. Swap in lunar/satellite-style images (OHRC/TMC-2) — mainly a preprocessing
   change (these images are often single-channel/grayscale, very different
   from natural photos, so the depth model may need domain adaptation or
   at least careful normalization).
3. Add camera path recording ("flythrough" mode) — scripted camera moves
   through the point cloud, optionally exportable as a video for a backup
   demo in case live interaction fails on stage.
4. Add basic mesh smoothing/hole-filling for cleaner surfaces instead of
   a raw point cloud, if time allows.