# Segmentation and instrument geometry

- `realtime_segment.py`: original camera/video segmentation, mask postprocessing, pixel-scale features and ZMQ publication.
- `realtime_improved_tip.py`: distal-region tip localization, open-jaw handling, temporal tracking, FOV estimation and specular-instance filtering. The head ROI classifier imports its geometry functions without starting a camera loop.
- `video_segment.py`: offline segmentation overlays.
- `calculate_fov.py`: optical aperture center/radius estimation.
- `ZMQ_Receiver_Plot.m`: MATLAB subscriber for tip, width and area plots.

Run a Python entry point with `--help` for options. Supply a trained segmentation checkpoint and your own video/camera input. The runtime produces image-space measurements, not calibrated metric depth. MATLAB requires JeroMQ separately.
