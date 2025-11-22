# Tooling Overview

This folder now groups maintenance scripts, ML utilities, diagnostics, and experimental helpers.

## Structure

- `maintenance/`: operational scripts for seeding, resetting, and backfilling the attendance database.
- `ml/`: training or embedding utilities used while iterating on face-recognition models.
- `diagnostics/`: quick scripts for checking services, running Windows console capture helpers, etc.
- `experiments/`: prototypes or throwaway POCs (e.g., DeepFace GUI) that are useful for reference but not part of production code.

Add new scripts under the correct subfolder and include a short usage comment at the top of the file. This keeps the root clean and makes it obvious what can be run safely in production vs. what is purely experimental.
