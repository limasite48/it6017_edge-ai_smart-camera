### Course: IT6017 - Technology trends in digital transformation
### Project Topic: Research and Reverse Engineering of Edge AI Camera Systems for Human Detection and Fire Detection.

! This project serves as the final assessment for the course IT6017.

/////////////////////////////////

**Teacher Requirements**
For the "Research and Reverse Engineer" type of project, teams are required to do the following:
    - Select a real-world Edge AI application (e.g., robot vacuum cleaners, smart cameras, smart wearables, automated incident detection, etc.)
    - Learn about hardware system configuration
    - Research and propose AI algorithms that can be implemented within the system
    - Evaluate and compare the options: performance, accuracy, energy consumption, cost, etc.
    - Write a detailed report + Presentation

**Teacher Notes**
The project must strictly adhere to the requirements for surveying a specific, real-world system and must demonstrate the feasibility of reverse engineering it.

/////////////////////////////////

**Anticipated components within the project**
1/ Hardware research (assigned to other team members):
    - Draw hardware architecture diagram.
    - Analyze the system's resource limitations.
    - Find information on the SoC, CPU, NPU, RAM, image sensor, power consumption, and cost.
    - Select 1 or 2 real-world AI camera systems to study.
2/ Researching AI algorithms (assigned to other team members):
    - Propose a model suitable for the target hardware. 
    - Investigate model formats, INT8 quantization, and compatibility with CPU/NPU execution. 
    - Research and compare 2–3 appropriate models, such as lightweight versions of YOLO or SSD MobileNet. 
    - Explore the pipeline for human detection and/or fire detection.
3/ Experimentation and evaluation (OUR WORK):
    - Analyze the limitations of the measurements and the differences between a smartphone and an AI-powered camera.
    - Explore the feasibility of deploying the model onto embedded hardware.
    - Collect data on latency, FPS, RAM usage, and accuracy (if labeled data is available).
    - Test the model on an Android smartphone or a laptop, if necessary.
4/ Reverse engineering and synthesis (assigned to other team members):
    - Consolidate reports and presentation slides, and assign tasks for presentation preparation.
    - Propose implementation plans, cost estimates, and optimization strategies.
    - Develop the overall hardware and software architectural design.
    - Analyze the processing pipeline: Camera → pre-processing → AI inference → post-processing → alert generation.

`We only need to complete section 3. The other sections are being handled by other team members.`

/////////////////////////////////

`Experimental hardware (testbed + benchmark): Android smartphone, model Xiaomi 12S Pro (actual device)`
    - SoC: Qualcomm Snapdragon 8+ Gen 1 (1x Cortex-X2 @ 3.19GHz + 3x Cortex-A710 @ 2.75GHz + 4x Cortex-A510 @ 1.80GHz).
    - Graphics / AI Accelerator: Adreno 730 GPU + Hexagon NPU (supported via Android NNAPI).
    - Memory: 8GB LPDDR5 RAM, 128GB/256GB UFS 3.1 storage.
    - Operating System: Android 13 (HyperOS / MIUI 14).
 
`Hardware for comparison: common consumer-grade cameras such as the Xiaomi Smart Camera C300/C400, Wyze Cam v3, etc. (only study, don't have actual device)`
    - SoC: Ingenic T31.
    - AI Accelerator (NPU): 0.5 TOPS
    - Memory: 128 MB DDR3 RAM (shared across the OS, H.264/H.265 video decoding streams, and AI models); 16–32 MB SPI NAND/NOR Flash ROM.
    - Image Sensor: 3MP/2K CMOS sensor (OmniVision OS04B10 / GalaxyCore) with a MIPI-CSI interface.
    - Power & Thermal Management: DC 5V/1A (approx. 5W power consumption); fully passive cooling (fanless) within a sealed plastic enclosure.