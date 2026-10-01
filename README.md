# Live Aero Fit

Local webcam app for estimating frontal area and CdA from a time-trial position.

## Run

```powershell
.\.venv\Scripts\python.exe app.py --port 8000
```

Open `http://localhost:8000`.

Open `http://localhost:8000/web/calculator.html` for the AeroTT CdA Lab. It
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

The classifier panel uses the same YOLO segmentation pass to draw live bounding
boxes and confidence labels, then writes a short natural-language summary from
the current detections and aero estimates.
