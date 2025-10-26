# JDWebcam

JDWebcam turns a regular webcam into a lightweight Just Dance virtual controller. It streams motion data to the game using body tracking instead of a phone accelerometer. Best suited for laptop users playing Just Dance 2017 (PC). Console support is technically possible but not implemented.

Note: scoring is less precise than the official controller app since this is not a real accelerometer. The goal is accessibility and experimentation rather than competitive accuracy.

## Requirements

* Just Dance 2017
* [Python 3.12](https://www.python.org/downloads/release/python-31210/)
* main.py

## Installation

On windows, go to Settings > Bluetooth & devices > Cameras, select your camera, and click Edit under "Advanced camera options". Toggle on **"Allow multiple apps to use the camera at the same time"**.

Then run this on cmd:

```
pip install websockets opencv-python mediapipe numpy
```

```
pip install --upgrade protobuf
```

Open Just Dance's "play with your smartphone" screen.

Double-click or start the program with:

```
python main.py
```

## FAQ and workarounds

* Can't change dancers -> connect a phone before
* Console not supported -> should work by tweaking a bit of code
* Scoring isn't very accurate -> tap your shoulder when only scoring OKs
* Multiplayer isn't implemented -> possible if you have multiple laptops on the same network

If you're a dev, feel free to implement things, fix bugs, commit changes and contact me.

## License

MIT License
