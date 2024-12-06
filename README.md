# CNN-Based Symbol Decoder for LoRa Uplink

This project implements a Convolutional Neural Network (CNN) to perform decoding of LoRa symbols. The test/training generation is a simulation of the LoRa modulation/demodulation with an asynchronous spreading factor collision across a complex Gaussian channel. The propagation model samples the number collisions as a Poisson process for increasing rate parameters.

Two methods of preprocessing on the complex-valued samples are tested. One with the FFT data being converted to binary images used as a 2D input for the CNN, where different methods of scaling are examined. Additionally, the direct use of IQ data is implemented. The model architectures for the FFT-CNN and the IQ-CNN are derived from [1] and [2], respectively.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Model Architecture](#model-architecture)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Prerequisites
- PyTorch (For the CNN model)
- TensorFlow (For test/training simulation)

## Model Architectures
#### CNN - Frequency-domain Symbol Detector 
| Layer | Type             | Kernel size/stride | Output Channels | Output Shape  | Parameters  |
|-------|-------------------|--------------------|-----------------|---------------|-------------|
| 1     | Convolution      | 4×4/1             | 32              | 129×129       | 544         |
| 2     | Batch Normalization |                    | 32              | 129×129       | 64          |
| 3     | Average Pooling  | 2×2/2             | 32              | 64×64         | 0           |
| 4     | Convolution      | 4×4/1             | 64              | 65×65         | 32,832      |
| 5     | Batch Normalization |                    | 64              | 65×65         | 128         |
| 6     | Average Pooling  | 2×2/2             | 64              | 32×32         | 0           |
| 7     | Flatten          |                    | 65536           |               | 0           |
| 8     | Fully Connected  |                    | 512             |               | 33,554,944  |
| 9     | Batch Normalization |                    | 512             |               | 1,024       |
| 10    | Fully Connected  |                    | 256             |               | 131,328     |
| 11    | Batch Normalization |                    | 256             |               | 512         |
| 12    | Fully Connected  |                    | 128             |               | 32,896      |

#### CNN - Time-domain Symbol Detector
| Layer | Type               | Kernel size (stride) | Output Channels | Output Shape | Parameters |
|-------|--------------------|-----------------------|-----------------|--------------|------------|
| 1     | Convolution 1      | 1 × 7 (1)            | 64              | 2 × 128      | 512        |
| 2     | Batch Normalization|                       |                 | 2 × 128      | 128        |
| 3     | ReLU               |                       |                 | 2 × 128      | 0          |
| 4     | Max Pooling        | 1 × 2 (2)            |                 | 2 × 64       | 0          |
| 5     | Convolution 2      | 1 × 7 (1)            | 128             | 2 × 64       | 57,472     |
| 6     | Batch Normalization|                       |                 | 2 × 64       | 256        |
| 7     | Max Pooling        | 1 × 2 (2)            |                 | 2 × 32       | 0          |
| 8     | Convolution 3      | 1 × 7 (1)            | 256             | 2 × 32       | 229,632    |
| 9     | Batch Normalization|                       |                 | 2 × 32       | 512        |
| 10    | Max Pooling        | 1 × 2 (2)            |                 | 2 × 16       | 0          |
| 11    | Flatten            |                       |                 | 8192         | 0          |
| 12    | Dropout (0.1)      |                       |                 | 8192         | 0          |
| 13    | Fully Connected    |                       | 128             |              | 1,048,704  |


## Contact 
Contact [cnielo21@student.aau.dk](mailto:cnielo21@student.aau.dk) for questions regarding the project.

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgements
- Supervisors Troels Bundgaard Sørensen & Poonam Maurya
- [AAU HPC Services](https://hpc.aau.dk/ai-lab/)

## References
1. Angesom Ataklity Tesfay et al. “Deep Learning-based Signal Detection for Uplink in LoRa-like Networks”. In: 2021 IEEE 32nd Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC). ISSN: 2166-9589. Sept. 2021, pp. 617–621. doi: 10.1109/PIMRC50174.2021.9569470. url: https://ieeexplore.ieee.org/document/9569470
2. Kosta Dakic et al. “LoRa Signal Demodulation Using Deep Learning, a Time-Domain Approach”. en. In: 2021 IEEE 93rd Vehicular Technology Conference (VTC2021-Spring). Helsinki, Finland: IEEE, Apr. 2021, pp. 1–6. isbn: 978-1-72818-964-2. doi: 10.1109/VTC2021-Spring51267.2021.9448711. url: https://ieeexplore.ieee.org/document/9448711/

