# JDWebcam

JDWebcam turns a regular webcam into a lightweight Just Dance virtual controller. It streams motion data to the game using body tracking instead of a phone accelerometer. Best suited for laptop users playing Just Dance 2017 (PC). Console support is technically possible but not implemented.

Note: scoring is less precise than the official controller app, since this does not emulate a real accelerometer. The goal is accessibility and experimentation rather than competitive accuracy.

## Requirements

* Just Dance 2017 (PC)
* Python 3.9+

## Installation

After installing Python, run this on cmd:

```
pip install websockets opencv-python mediapipe numpy
```

Then double-click or start the program with:

```
python main.py
```

## License

MIT License
