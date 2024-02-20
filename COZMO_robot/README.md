# COZMO SDK

# Installation

- Initial setup

To use the Cozmo SDK, the Cozmo mobile app must be installed on your mobile device and that device must be tethered to a computer via USB cable.

- Prerequisite

Python 3.5.1 or later

WiFi connection

An iOS or Android mobile device with the Cozmo app installed, connected to the computer via USB cable

- SDK examples programs

Anki provides example programs for novice and advanced users to run with the SDK. Download the SDK example programs.

- Installation 

For Windows : 

To install the SDK, type the following into the Command Prompt window:

```
pip install --user cozmo[camera]
```

The [camera] option adds support for processing images from Cozmo’s camera.

To upgrade the SDK from a previous install, enter this command:

``` 
pip3 install --user --upgrade cozmo
```

Mobile device setup : 

iOS devices require iTunes to ensure that the usbmuxd service is installed on your computer. Usbmuxd is required for the computer to communicate with the iOS device over a USB cable. While iTunes needs to be installed, it does not need to be running.

Android devices require installation of Android Debug Bridge (adb) in order to run the Cozmo SDK. This is required for the computer to communicate with the Android mobile device over a USB cable and runs automatically when required.

- Starting up the SDK

1. Plug the mobile device containing the Cozmo app into your computer.

2. Open the Cozmo app on the phone. Make sure Cozmo is on and connected to the app via WiFi.

3. Tap on the gear icon at the top right corner to open the Settings menu.

4. Swipe left to show the Cozmo SDK option and tap the Enable SDK button.

5. Make sure the SDK examples are downloaded from the Downloads page.

6. On the computer, open Terminal (macOS/Linux) or Command Prompt (Windows) and type cd cozmo_sdk_examples, where cozmo_sdk_examples is the directory you extracted the examples into, and press Enter.

- Run a program 

To run a program, using the same Terminal (macOS/Linux) / Command Prompt (Windows) window mentioned above:

1. cd into the folder where your code is (careful : for Windows, there must have no '/' but '\')

2. type the following and press Enter : 
``` 
python <your_code>.py
```

## How to connect to COZMO 

Install pycozmo on your computer

Make sure COZMO is sitting on its recharging base

Plug it into a USB power supply on your computer => the green indicator on COZMO shows that it is charging 

Connect the computer to the wifi of the COZMO robot (remember to keep a pre-printed copy of your reference notes and programming codes because you will not be able to access internet while connected to COZMO)

Go to your the place permitting to connect to wifi on your computer 

Search for the COZMO wifi in the available wifis

To show the password, tap on the button on the top of the COZMO, the password will appear on the screen of the COZMO

Go to your python program on your computer (for example on VS CODE) 

Click on ‘Run Python File in Terminal’

Remember : to run the codes correctly don’t forget to import everything necessary 

Basics are : import cozmo, import pycozmo

for each utilitarian : from cozmo.util import utilitarian

