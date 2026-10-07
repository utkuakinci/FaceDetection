# Face Detection with Python

This project demonstrates real-time face detection and emotion recognition with Python, using a webcam:

- Faces are detected with OpenCV Haar Cascades.

- The emotion of each detected face is predicted with the [fer](https://github.com/justinshenk/fer) library.

- A box and a label such as `happy (87%)` are drawn on each face in real time.

## 📦 Requirements

- Python version supported by TensorFlow

- OpenCV 4.x (`opencv-contrib-python<5`)

- fer (installs TensorFlow)

# Installation

## Quick Start

You can install the FaceDetection environment by using the following commands:

```shell
$ git clone https://github.com/utkuakinci/FaceDetection.git
$ cd FaceDetection
$ pip install -r requirements.txt
```

# Usage

```shell
$ python main.py
```

Press `q` to quit.

# Tests

```shell
$ pip install -r requirements-dev.txt
$ pytest
```

The emotion tests use a stub in place of fer, so they do not need TensorFlow. To enable the sample-image test, add a photo of a face at `tests/data/sample_face.jpg`.
