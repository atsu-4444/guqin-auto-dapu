# Initial setup for music21 (Run this only once)
from music21 import environment

# Set the path to the MuseScore executable.
# This is a common default path for Windows. Adjust as necessary for your environment.
musescore_path = 'c:/Program Files/MuseScore 4/bin/MuseScore4.exe' 

us = environment.UserSettings()
us['musicxmlPath'] = musescore_path
us['musescoreDirectPNGPath'] = musescore_path

print("music21 setup complete.")