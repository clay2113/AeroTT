# Live Aero Fit

Local webcam app for estimating frontal area and CdA from a time-trial position.

## Run

```powershell
.\.venv\Scripts\python.exe app.py --port 8000
```

Open `http://localhost:8000`.

Open `http://localhost:8000/calculator.html` for the AeroTT CdA Lab. It
lists the working calculations on the left: CdA estimate, air density, aero
power, rolling power, wheel power, CdA from power, and speed from power. Each
feature updates from the ride inputs and shows its formula below the result.
The latest CdA produced by the camera page is saved locally and can be loaded
with `Use latest camera value`.

Use `Start camera` for live webcam analysis, or `Upload pic` to analyze a still
image through the same segmentation, classifier, frontal area, CdA, and
description pipeline. After changing calibration or threshold settings on a still
image, click `Analyze pic` to rerun the estimate.

## Calibration

The default mode uses the live mask width as the pixel reference and treats the
known width as `0.42 m`. For better numbers, turn off auto calibration, click
`Set points`, then click two points in the original video that correspond to the
known width you entered.

CdA is estimated as `frontal area * Cd multiplier`; the camera can estimate the
area, but Cd is still an aerodynamic assumption unless you combine this with
power, speed, air density, and rolling resistance data.

## Deploy

The frontend is hosted by Firebase Hosting. Hosting forwards `/api/**` to the
Cloud Run service `aerott-api`, so the browser uses one origin and no separate
CORS configuration is needed. The Cloud Run image includes the CPU PyTorch
runtime and `yolov8n-seg.pt`; the SAM checkpoint and training data are not used.

Create a Firebase project with Hosting enabled. Install the Google Cloud CLI,
Node.js, and the Firebase CLI (`npm install -g firebase-tools`), then
authenticate and select your Google Cloud project:

```powershell
gcloud auth login
firebase login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

From the repository root, build and deploy the backend first:

```powershell
gcloud run deploy aerott-api `
	--source . `
	--region us-central1 `
	--allow-unauthenticated `
	--memory 4Gi `
	--cpu 2 `
	--concurrency 1 `
	--timeout 300 `
	--max-instances 5
```

Then publish the static frontend from `web/`:

```powershell
firebase deploy --only hosting --project YOUR_PROJECT_ID
```

The Firebase rewrite in `firebase.json` must use the same project as the Cloud
Run service. Cloud Run is publicly invokable so Firebase Hosting can forward
the API requests; set billing alerts and review access/cost requirements before
using this for a public deployment. The first request after scale-to-zero can
take longer while the model starts.

The classifier panel uses the same YOLO segmentation pass to draw live bounding
boxes and confidence labels, then writes a short natural-language summary from
the current detections and aero estimates.
