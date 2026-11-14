# math-matrix: the App Lab app

The demo face of the copilot: a web page served by the board, and the board's 8x13 LED matrix.
It is separate from `predict.py`, which the jury runs from the terminal.

## What it does

- **Write text:** up to 2 characters stay still; longer text scrolls (A–Z, 0–9, `! ? . , - + = : *`).
- **Draw pixels:** paint the 8x13 grid on the page and send it to the matrix.
- **Answer A–E:** shows `?`, checks the model server, then shows the letter. `work()` in
  `python/main.py` is where copilot logic (hints, checking answers) goes.

## Layout

```text
app.yaml                 name and bricks (arduino:web_ui)
python/main.py           runs on Linux in App Lab's container: web API, model calls, matrix frames
python/setup_bridge.sh   one-time board setup so the app can reach the model (see below)
assets/index.html        the web page (port 7000)
sketch/sketch.ino        runs on the microcontroller: draws frames received over the Bridge
sketch/sketch.yaml       board platform
```

## Install on a board

1. Copy this folder to `/home/arduino/ArduinoApps/math-matrix/` on the board, or create an app
   called `math-matrix` in App Lab, add the **WebUI - HTML** brick, and paste in the files.
2. Run it from App Lab (or `arduino-app-cli app start user:math-matrix`).
3. Open `http://<board-ip>:7000`, or over USB: `adb forward tcp:7000 tcp:7000` then `http://127.0.0.1:7000`.

## Reaching the model from the app

App Lab runs `main.py` inside a Docker container, where `127.0.0.1` is the container, not the board,
so the app can't reach the model servers (`math-llm` on 127.0.0.1:8081) directly.

- `main.py` finds the board's address as the container's default gateway (for example `172.19.0.1`).
- `python/setup_bridge.sh` installs the `llm-bridge` service, which forwards port 8081 on each Docker
  network gateway to `127.0.0.1:8081`. It listens only on the Docker networks, not on Wi-Fi, and
  starts at boot. Run it once per board:

  ```bash
  sudo bash ~/ArduinoApps/math-matrix/python/setup_bridge.sh
  ```

Check with `systemctl is-active llm-bridge` and `ss -ltn | grep 8081`.
