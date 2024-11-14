# CNN Symbol Decoder for LoRa Uplink

This project implements a Convolutional Neural Network (CNN) to perform decoding of LoRa symbols. The test/training generation is a simulation of the LoRa modulation/demodulation with an asynchronous spreading factor collision across a complex Gaussian channel. The propagation model samples the number collisions as a Poisson process for increasing rate parameters.

Two methods of preprocessing on the complex-valued samples are tested. One with the FFT data being converted to non-scaled binary images used as a 2D input for the CNN and another with the direct use of the IQ data.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Model Architecture](#model-architecture)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Prerequisites
- PyTorch (For the CNN model)
- TensorFlow (For test/training simulation)



## Model Architecture
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

## Contact 
Contact [cnielo21@student.aau.dk](mailto:cnielo21@student.aau.dk) for questions regarding the project.

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgements
- Supervisors Troels Bundgaard Sørensen & Poonam Maurya
- [AAU HPC Services](https://hpc.aau.dk/ai-lab/)
