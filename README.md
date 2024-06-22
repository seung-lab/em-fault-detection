# Electron Microscopy Fault Detection

Detecting issues via online image analysis during high throughput electron microscopy. 

The goal is to detect high consequence errors during imaging that could disrupt the rest of the reel. For example, large tears in the image.

The system used is a Kapton tape fed Transmission Electron Microscope (TEM) that provides both low magnification overview snapshots of the sample as well as high resolution tiles.

Broadly we are attempting to detect three categories: 

1. HALT: Immediately halts automated imaging and waits for manual intervention.
2. WARNING: Notifies the microscopist or logs the issue.
3. OK: Proceed with imaging.

The work will begin with the analysis of a run of low magnification images.



