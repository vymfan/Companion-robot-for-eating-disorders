# Instructions on how to install DepthAI and use the OAK-D camera

## Installing DepthAI

- Follow instructions below to install DepthAI and its dependencies/requirements with an installer : 
Windows 10/11 users can install DepthAI with the Windows Installer. Find it on https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#default-run 
Installer will install either the newer DepthAI Viewer (visualization GUI application), or DepthAI Demo (python script, older GUI application) or and all the dependencies. We suggest using the DepthAI Viewer.
After the installer finishes, you can directly run the DepthAI app from the list of applications, which will run the installed demo.

- If the installer doesn’t work due to your antivirus, desactivate this one by going in Task bar > Antivirus > Desactivate for 10 minutes 

- If it still doesn’t work or you prefer installing manually, follow the next steps :
Create a new environment in Anaconda. Make sure to put a package python > 3.8 
Python 3.11 might be to evaluate on some computer also. 
Python 3.9 seems to be a good alternative.

Go in the Control Window or Anaconda Prompt on your computer.

-- First you need to clone the github repository with
'''bash
git clone --recursive https://github.com/luxonis/depthai.git
'''
Remarq : In case you have already cloned the repository, you can update your submodules with:
'''bash
git pull --recurse-submodules
'''

1. There are two installation steps that need to be performed to make sure the demo works:

1.1. One-time installation that will add all necessary packages:
For Windows 10/11, we recommend using the Chocolatey package manager to install DepthAI’s dependencies on Windows. Chocolatey is very similar to Homebrew for macOS.
To install Chocolatey and use it to install DepthAI’s dependencies do the following:
Right click on Start on computer 
Choose Windows PowerShell (Admin) and run the following:
'''bash 
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
'''
Close the PowerShell and then re-open another PowerShell (Admin) by repeating the first two steps.
Install Python and PyCharm
'''bash
choco install cmake git python pycharm-community -y
'''

1.2. Python dependencies installation that makes sure your Python interpreter has all required packages installed. This script is safe to be run multiple times and should be ran after every demo update
'''bash 
python install_requirements.py
'''

- Installing Depthai viewer : 
'''bash
python -m pip install depthai-viewer
'''

Launch : 
'''bash 
depthai-viewer
'''

