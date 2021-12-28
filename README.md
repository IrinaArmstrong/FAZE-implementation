# FAZE: Few-Shot Adaptive Gaze Estimation 
## Implementation as Gaze Estimation Tool
Upgraded and slightly changed FAZE framework implementation, hardly based on original paper[^1] and authors [source code](https://github.com/NVlabs/few_shot_gaze).

## General Information
FAZE is a framework for few-shot adaptation of gaze estimation networks, consisting of equivariance learning 
(via the DT-ED or Disentangling Transforming Encoder-Decoder architecture) and meta-learning with gaze-direction embeddings as input.
In this repository I adopted FAZE for the [GazeCapture](https://github.com/CSAILVision/GazeCapture)[^2] dataset.

## Performance & Usage
The code is intended to be used as a tool for preprocessing raw video frames obtained directly from a user's webcam in the process of stimulating eye movements. 
The output data are the directions of a person's gaze, which are then applied in a more complex task of **the viewer's personality identification**.

## Prerequisites
This version of the code is *adapted to run on Windows* (developed on Windows 10) and has an optional *ability to run on both GPU and CPU*.
All required Python package dependencies can be installed by running from pypy/conda:
````bash 
pip install -r requirements.txt
````

## Links
[^1] Park, S., Mello, S.D., Molchanov, P., Iqbal, U., Hilliges, O., & Kautz, J. (2019). Few-Shot Adaptive Gaze Estimation. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 9367-9376.
[^2] Krafka, K., Khosla, A., Kellnhofer, P., Kannan, H., Bhandarkar, S.M., Matusik, W., & Torralba, A. (2016). Eye Tracking for Everyone. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2176-2184.
