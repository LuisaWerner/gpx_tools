# GPX Elevation Profile and Track Renderer

An automated Python tool that converts a `.gpx` route file into an animated, high-fidelity elevation profile video. The output is rendered with a native alpha channel (transparency) using the **Apple ProRes 4444** codec inside a QuickTime `.mov` container.

This allows you to drag-and-drop the generated animation straight onto your video editing timeline (Premiere Pro, DaVinci Resolve, Final Cut Pro, CapCut, etc.) without having to deal with black backgrounds or chroma-key green screens.

## Features

- **True Alpha Transparency:** Zero background pixels (`0,0,0,0`) out of the box. No video blending mode tweaks required.
- **Dynamic Pacing & Easing:** Built-in easing curves ensure the drawing movement accelerates and slows down naturally.
- **Marker Synchronization:** Automatically scans custom geographic coordinates (lat/long) and pops up fixed indicator dots on the profile graph exactly when the animation line passes over those milestones.
- **Edge-to-Edge Fitting:** Designed to auto-scale horizontally from pixel `0` to `1080` to perfectly wrap cinematic vertically-oriented layouts ($1080 \times 1920$).
- **Dramatic Vertical Scaling:** Exaggerates peaks and climbs smoothly across the lower third of the frame for striking visual feedback.

## Prerequisites

Before running the script, you must have **FFmpeg** installed on your system and configured in your system's PATH.

- **macOS (via Homebrew):** `brew install ffmpeg`
- **Windows / Linux:** Download the binaries directly from the [FFmpeg Official Website](https://ffmpeg.org/download.html).

## Installation

1. Clone the repository or navigate to your local directory.
2. Install the Python dependencies using `pip`:

```bash
pip install -r requirements.txt