# Dataset and model notices

## RealWaste

WasteOps provides a downloader for the official UCI Machine Learning Repository record (dataset ID 908): https://archive.ics.uci.edu/dataset/908/realwaste. UCI currently lists the dataset as CC BY 4.0. Cite:

Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information*, 14(12), 633. https://doi.org/10.3390/info14120633.

A GitHub mirror has described its copy as CC BY-NC-SA 4.0. To avoid relying on a mirror with conflicting terms, the downloader uses the official UCI archive URL. Re-check the official record and image rights before commercial redistribution or model commercialization. Dataset files are not included in this repository.

## CLIP

The default zero-shot backend uses `openai/clip-vit-base-patch32` from Hugging Face. Model card: https://huggingface.co/openai/clip-vit-base-patch32. The model weights are downloaded at runtime and are not included in this repository. Review upstream model terms and usage limitations before commercial deployment.

## Third-party data

The MIT license in this repository applies only to original WasteOps source code. It does not grant rights to third-party datasets, model weights, trademarks, or images.
