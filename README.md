# Rice Variety Classification System

## Overview
This system classifies different varieties of rice based on their images. It uses a pre-trained MobileNet V2 model fine-tuned on a specialized dataset, providing accurate and efficient predictions for rice variety identification.

## Dataset Information
- **Source:** Dr. Murat Koklu’s research
- **Size:** 75,000 images total, with 15,000 images per variety
- **Rice Varieties:**
  - Arborio
  - Basmati
  - Ipsala
  - Jasmine
  - Karacadag

## Model Details
- **Architecture:** MobileNet V2 (pre-trained on ImageNet)
- **Purpose:** Classifies rice images into one of five varieties
- **Training:** Fine-tuned on the rice dataset to optimize for this specific task

## Key Features
- **Input:** Image of rice grains
- **Output:** Predicted rice variety
- **Performance:** Achieves high accuracy due to transfer learning and robust dataset preparation
- **User Options:**
  - Upload an image of rice
  - Use a live camera feed to capture an image of rice

## Technology Used
- **Framework:** The system is built using Streamlit, enabling an interactive and user-friendly interface.

## Acknowledgments
- Special thanks to Dr. Murat Koklu for providing the dataset.

