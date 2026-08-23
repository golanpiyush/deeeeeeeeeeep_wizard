# DepthWizard — how it works (plain English, file by file)

This is a walkthrough of every file in the project, written so anyone on
the team can read it out loud in a presentation or use it to answer a
judge's question on the spot. Files are ordered the way data actually
flows through the app.

---

## The one-sentence version

**Upload a photo → an AI guesses how far away every pixel is → we turn
that into a floating cloud of 3D points you can rotate → for real lunar
photos specifically, you can also click two points to get a genuine,
physics-based height measurement in meters.**

Two features, one app: a visual demo (looks impressive) and a science
tool (is scientifically defensible). Lead with the visual, then pivot to
"but here's the part that's actually rigorous" — that's the strongest
story for judges.

---

## `model_switch.py` — pick the AI model size

Depth Anything V2, the AI model that estimates depth, comes in three
sizes: small, base, large. Think of it like t-shirt sizes for the same
brain — small is fast and lower quality, large is slow but sharpest.

One line at the top of this file, `ACTIVE_MODEL = "small"`, controls
which size the whole app uses. Change it, restart the server, done. The
file also prints a colorful summary to the terminal so whoever's running
it can see at a glance what's active and what it costs (download size,
GPU memory needed, quality level).

**Say to judges:** "We can trade speed for quality with one line — small
model for fast iteration during development, large model for the polished
demo run on the GPU."

---

## `progress_events.py` — the messenger

Depth estimation and point-cloud building take real time — seconds, not
milliseconds. Without this file, the user would stare at a blank spinner
the whole time.

This file is a tiny notification system. Any slow part of the backend can
call `reporter.emit("some_event", "a message")` — like calling `print()`,
except instead of printing to a terminal, the message gets pushed live to
the browser over the WebSocket connection. That's what powers the glowing
green "live terminal" panel in the UI — it's not fake, it's real internal
status being streamed out as it happens.

If nobody's listening (no WebSocket connected), `emit()` just does
nothing — silently safe to call from anywhere, including scripts and
tests that don't have a live connection.

**Say to judges:** "The live terminal isn't decoration — it's genuinely
streaming internal pipeline state in real time, which is also great for
debugging."

---

## `depth_model.py` — the AI brain

This file wraps Depth Anything V2 so the rest of the app doesn't need to
know anything about PyTorch or Hugging Face internals.

What it does, in order:
1. Loads the model **once** when the server starts (loading is slow —
   a few seconds to a minute — so it's reused for every request instead
   of reloading each time).
2. Detects whether a GPU is available and uses it if so; falls back to
   CPU (much slower) otherwise, with a clear warning either way.
3. Takes a photo in, returns a "depth map" out — a 2D grid of numbers
   the same size as the image, where each number is the model's guess at
   how close or far that pixel is. This is **relative** depth (which
   things are nearer than others), not an exact real-world distance in
   meters.

**Say to judges:** "This is the same off-the-shelf Depth Anything V2
model anyone can use — the value we add isn't the model, it's everything
we built around it, especially the real physics measurement later."

---

## `pointcloud.py` — turning depth into a 3D shape

Every pixel in the photo has a flat (x, y) position. The depth map from
`depth_model.py` gives each pixel a z (how deep). This file combines
them: place a colored dot at (x, y, z) for each pixel, and you get a
"3D photo" — a point cloud — that can be rotated and flown through in
the browser.

A full photo can have millions of pixels, which is too much data to send
over the network and too much for a browser to render smoothly. So this
file randomly downsamples down to a capped number of points (currently
150,000) before sending the result.

It also builds the point cloud in **chunks** rather than all at once,
sending each chunk as a live progress event — this is what makes the
point cloud visibly build up on screen piece by piece, rather than
popping in all at once at the end.

**Say to judges:** "150,000 points is a deliberate trade-off — enough to
look genuinely detailed, low enough to stay smooth on a judge's laptop or
phone."

---

## `server.py` — the web server that ties it together

This is the FastAPI backend — the actual program that's running and
listening for requests. It exposes:

- **`GET /`** — a health check. Visiting it in a browser shows whether
  the server is up and which model is active.
- **`POST /predict`** — takes an uploaded photo, runs it through
  `depth_model.py` then `pointcloud.py`, and returns the finished point
  cloud in one response. No live updates here — just wait, then get the
  full result.
- **`WS /ws/predict`** — the WebSocket version. This is what the actual
  frontend uses. The browser opens a persistent connection, sends one
  image, and then receives a *stream* of live messages as the pipeline
  runs (model loading, inference timing, point-cloud chunks) followed by
  a final "result" message with the full point cloud. This is what
  powers the live terminal panel.
- **`POST /measure_height`** — the lunar shadow-height tool's endpoint.
  Takes two pixel coordinates plus an image's sun angle and resolution,
  and returns a real height measurement (see `shadow_height.py` below).

Depth estimation is CPU/GPU-heavy blocking work, which would normally
freeze the whole server while it runs. This file runs that work on a
background thread (`asyncio.to_thread`) so the server can keep talking to
the browser (sending live progress messages) at the same time the model
is actually computing.

**Say to judges:** "The WebSocket isn't just for show — it's the
mechanism that lets us show real, live internal state instead of a fake
loading bar."

---

## `shadow_height.py` — the actual scientific differentiator

Everything above produces a visually convincing but **scientifically
unvalidated** point cloud — the depth values are relative guesses from a
model trained on ordinary Earth photos, not real lunar measurements.

This file is different. It uses classical shadow photogrammetry — real
physics, no AI — to calculate an actual height in meters:

```
height = shadow_length_in_meters × tan(sun_elevation_angle)
```

A tall feature under a low sun casts a long shadow; a short feature casts
a short one. If you know the sun's angle (from the image's own metadata)
and measure the shadow's length in meters (pixel distance × the image's
real ground resolution), basic trigonometry gives you the feature's real
height. No machine learning involved — this is the same principle
surveyors and early astronomers used centuries ago.

Two things make this defensible rather than a toy:
- **Honest uncertainty**: it doesn't return a fake-precise number. It
  assumes the user's click could be off by a few pixels and calculates
  how much that swings the final answer, reporting a real ± range.
- **Honest failure**: if the sun angle is too close to the horizon (near
  0°), the math becomes unstable — tiny click errors turn into huge
  height errors. Instead of silently returning a misleading number, it
  flags the result as unreliable and explains why.

**Say to judges:** "This is the one number in the whole app that's
physically grounded and independently verifiable — not a neural network's
guess. We built in the honesty about when the sun angle makes it
unreliable on purpose, because a tool that hides its own limitations is
worse than one that states them."

---

## `index.html` — everything the user actually sees

One self-contained file: HTML, CSS, and JavaScript together. No build
step, no separate framework — open it and it runs.

It has two modes, switched with a toggle at the top:

**3D View mode** (the original app):
- Upload a photo, click Generate, and watch a live green terminal panel
  narrate the pipeline (model loading → inference → point-cloud chunks)
  while a Three.js 3D scene renders the result. Drag to rotate, pinch to
  zoom, works on both desktop and mobile with on-screen arrow controls.

**Lunar Measure mode** (the new differentiator UI):
- The center upload panel hides, and a lunar-specific panel appears at
  the top instead.
- Upload a lunar image crop, then click two points directly on it: the
  top of a crater rim or ridge, then the tip of its shadow.
- The app converts your on-screen clicks into the image's real original
  pixel coordinates (important — the image is scaled to fit the screen,
  but the math needs true pixel distances), sends them to
  `/measure_height`, and displays the real height result with its
  uncertainty and any reliability warning.

Both modes talk to the backend over a Cloudflare tunnel (`wss://` for the
WebSocket, regular `https://` for the height endpoint), which is what
lets teammates test the whole thing from their own phones and laptops
without needing the backend running locally.

**Say to judges:** "One HTML file, two complete workflows — the visual
wow factor and the scientific tool, switchable with one click."

---

## `generate_sample_depth.py` — the fake mode (dev tool only)

Not part of the real demo. This exists purely so you can test the *rest*
of the pipeline (server, point cloud, 3D viewer, live terminal) in
seconds, without waiting for the real AI model to load or download.

It fakes a "depth map" using nothing but classical image processing:
convert to grayscale, blur it, and treat brightness as a crude stand-in
for depth (brighter = closer). It is explicitly **not** a real depth
estimate — just a fast placeholder for plumbing-testing. Turn it on with
the `USE_FAKE_DEPTH=1` environment variable.

**Say to judges (if asked):** "This is a development tool, not part of
the actual pipeline — it let us build and test the rest of the app before
the real model was fully wired in."

---

## How it all connects, start to finish

1. User opens `index.html` in a browser.
2. **3D View path:** photo → `server.py` (`/ws/predict`) → `depth_model.py`
   → `pointcloud.py` → live progress back to the browser the whole time
   → final point cloud rendered in Three.js.
3. **Lunar Measure path:** lunar image → two clicks → `server.py`
   (`/measure_height`) → `shadow_height.py` → real height + uncertainty
   shown in the panel.
4. `model_switch.py` and `progress_events.py` are support files used by
   both paths — one picks the AI model size, the other carries live
   status messages.
5. `generate_sample_depth.py` is a dev-only shortcut, not part of the
   real demo flow.

The pitch in one line: **most teams will show a point cloud. We show a
point cloud *and* a real, physically-grounded measurement with honest
uncertainty — because that's what separates a cool visualization from a
tool ISRO could actually trust.**