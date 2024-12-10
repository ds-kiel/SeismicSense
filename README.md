

# SeismicSense: Earthquake Event Detection and Analysis

This repository contains the codebase for **SeismicSense**, a tool designed for detecting and analyzing earthquake events, focusing on **P** and **S phase arrivals**. 

## Dataset
The model utilizes the [STEAD Dataset](https://github.com/smousavi05/STEAD.git), a curated dataset for seismic event detection.

## Workflow

### Preprocessing and Training
1. **Preprocessing and Training**  
   Start with the `SeismicSense_preprocess_n_train.py` script:
   - Set `mode="prepare"` to preprocess the data.
   - Once preprocessing is complete, set `mode="train"` to train the model.

2. **Testing the Original Model**  
   Use `SeismicSense_test.py` to evaluate the trained model on the test set.

### Model Quantization
1. **Quantize the Entire Model**  
   Use `SeismicSensequant.py` to quantize the trained model.
   - Test the quantized model with `SeismicSensequant_test.py`.

2. **Split Models for Quantization**  
   If you prefer to work with split models:
   - Use `Split.py` to split the model.
   - Quantize the split models using `SeismicSensequant_split.py`.
   - Test the quantized split models with `TestingSplit_quant.py`.

## Repository Overview
- `SeismicSense_preprocess_n_train.py`: Handles data preparation and model training.
- `SeismicSense_test.py`: Tests the original trained model.
- `SeismicSensequant.py`: Quantizes the entire model.
- `SeismicSensequant_test.py`: Tests the fully quantized model.
- `Split.py`: Splits the model into parts for modular handling.
- `SeismicSensequant_split.py`: Quantizes the split models.
- `TestingSplit_quant.py`: Tests the quantized split models.

## Getting Started
1. Clone this repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the workflow steps above for your use case.

## Acknowledgment
The **STEAD Dataset**: [STEAD GitHub Repository](https://github.com/smousavi05/STEAD.git)
