# SuPreM Demo

This repository contains a demo website for processing CT scans and obtaining results using the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) AI model. The demo is hosted on Vercel and leverages external GPU resources to run the AI inference.

You can view the live demo here: [TotalSegmentator Demo](https://su-pre-m-demo-8n9pbxhkz-ethan-lees-projects-51d2fc73.vercel.app/).

## Project Overview

The goal of this project was to create a functional web demo that processes CT scan data and performs segmentation using AI models. The primary AI model resource, `inference.py`, along with other related files, was retrieved from a Docker image available on Docker Hub. These resources were adapted and integrated into the web platform for streamlined usage.

### Key Features:
- **AI-Powered Segmentation**: Utilizes the TotalSegmentator AI model to perform segmentation on uploaded CT scan data.
- **Cloud-Based GPU Execution**: Since the development environment lacked an NVIDIA GPU, I utilized **RunPod** for handling GPU resources, load balancing, and infrastructure for AI inference.
- **Web Integration**: The website is hosted on **Vercel**, which manages the frontend, CDN, blob storage, and the necessary API connections for serving the segmentation results.

## Technical Workflow

1. **Model Resources**: AI model resources, including `inference.py`, were extracted from a Docker image hosted on Docker Hub.
2. **Infrastructure**: Since the local development environment lacked the necessary hardware for running AI models, **RunPod** was used to access cloud GPUs and manage the necessary infrastructure for running the TotalSegmentator model.
3. **Deployment**: The frontend was built with **Next.js** and hosted on **Vercel**. Vercel also manages other services like blob storage and APIs, ensuring smooth data handling and result generation.

<p>
  <div align="center">
    <img width="70%" src="https://github.com/AstroHyo/SuPreM_demo/blob/main/nextjs/public/SuPreM_demo_Screenshot.png">
  </div>
</p>
