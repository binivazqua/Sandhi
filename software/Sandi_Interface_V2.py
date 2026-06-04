#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on June 02, 2026, at 18:36
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from Serial_Begin
import serial
import os as _os
# Run 'Before Experiment' code from EEG_Start_Code
from eeg_lsl_bridge import verify_eeg_stream, SandhiMarkerOutlet, MARKERS
import time

_NO_HARDWARE = _os.environ.get('SANDHI_NO_HARDWARE', '0') == '1'

if _NO_HARDWARE:
    # Mock mode: bypass EEG stream and serial ports for UI testing without hardware.
    # Activate with:  SANDHI_NO_HARDWARE=1 python Sandi_Interface_V2.py
    print("[Sandhi] *** SANDHI_NO_HARDWARE=1 — running in mock mode, no EEG or serial ***")

    class _MockMarkerOutlet:
        def push(self, marker, verbose=True):
            if verbose:
                print(f"[Sandhi][MOCK] Marker: '{marker}'")

    class _MockSerial:
        in_waiting = 0
        def readline(self): return b''
        def close(self): pass

    marker_outlet = _MockMarkerOutlet()
    esp32   = _MockSerial()
    esp32_1 = _MockSerial()
else:
    verify_eeg_stream()      # aborta si no hay stream EEG
    marker_outlet = SandhiMarkerOutlet()

time.sleep(0.5)
# Run 'Before Experiment' code from code_2
import random
# Run 'Before Experiment' code from Timestamps2
import os
# Run 'Before Experiment' code from code_2
import random
# Run 'Before Experiment' code from code_2
import random
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'Sandi_Interface'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\gianl\\OneDrive\\Escritorio\\Sandhi_Demo\\Sandi_Interface.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Start_Exp" ---
    Indicaciones_5 = visual.TextStim(win=win, name='Indicaciones_5',
        text='¡Bienvenid@ a Sandhi Interface!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
        font='Arial',
        pos=(0, 0.30), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_5 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_5.mouseClock = core.Clock()
    play_5 = visual.ImageStim(
        win=win,
        name='play_5', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    # Run 'Begin Experiment' code from Serial_Begin
    if not _NO_HARDWARE:
        esp32   = serial.Serial('COM3', 115200, timeout=0.01)
        esp32_1 = serial.Serial('COM4', 115200, timeout=0.01)
    # Run 'Begin Experiment' code from EEG_Start_Code
    marker_outlet.push(MARKERS.BLOCK_START)
    print("EEG marker sent: BLOCK_START")
    
    # --- Initialize components for Routine "Instruc_Emotions" ---
    Indicaciones_6 = visual.TextStim(win=win, name='Indicaciones_6',
        text='• Tienes 3 segundos para identificar la emoción mostrada en la imagen.\n• Utiliza el slider para registrar tu respuesta.\nPresiona el botón verde para comenzar.\n',
        font='Arial',
        pos=(0, 0.30), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_6 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_6.mouseClock = core.Clock()
    play_6 = visual.ImageStim(
        win=win,
        name='play_6', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "Emotions_Trial" ---
    Instruccion_2 = visual.ImageStim(
        win=win,
        name='Instruccion_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    circulo = visual.ImageStim(
        win=win,
        name='circulo', 
        image='Assets/circulo_verde.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.32), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Cuenta_tarea3 = visual.TextStim(win=win, name='Cuenta_tarea3',
        text='',
        font='Arial',
        pos=(0, -0.32), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    respuesta_slider = visual.Slider(win=win, name='respuesta_slider',
        startValue=None, size=(1.0, 0.1), pos=(0, -0.15), units=win.units,
        labels=('1', '2', '3', '4'), ticks=(1, 2, 3, 4), granularity=0.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor=(-1.0000, -1.0000, -1.0000), markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    # Run 'Begin Experiment' code from Contador
    contador = 1; 
    
    # --- Initialize components for Routine "Black_Screen" ---
    Wait = visual.Rect(
        win=win, name='Wait',
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Start" ---
    Boton_Amarillo_Start = visual.ImageStim(
        win=win,
        name='Boton_Amarillo_Start', 
        image='Assets/boton_amarillo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.33, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Boton_Rojo_Start = visual.ImageStim(
        win=win,
        name='Boton_Rojo_Start', 
        image='Assets/boton_rojo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.54, -0.30), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Indicaciones = visual.TextStim(win=win, name='Indicaciones',
        text='¡Bienvenid@ a Sandhi Interface!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
        font='Arial',
        pos=(0, 0.36), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    play = visual.ImageStim(
        win=win,
        name='play', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "Instruc_Buttons" ---
    Boton_Amarillo_Start_2 = visual.ImageStim(
        win=win,
        name='Boton_Amarillo_Start_2', 
        image='Assets/boton_amarillo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.33, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Boton_Rojo_Start_2 = visual.ImageStim(
        win=win,
        name='Boton_Rojo_Start_2', 
        image='Assets/boton_rojo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.54, -0.30), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Indicaciones_2 = visual.TextStim(win=win, name='Indicaciones_2',
        text='• En la botonera presiona el botón del color que aparece en pantalla.\n\n• Si sale un color distinto al amarillo o al rojo, no deberás presionar ningún botón.\n\n•Tienes 1.5s para seleccionar el botón\n\nPresiona el botón verde para empezar',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    play_2 = visual.ImageStim(
        win=win,
        name='play_2', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.20), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "Buttons_Trial" ---
    Boton_Amarillo = visual.ImageStim(
        win=win,
        name='Boton_Amarillo', 
        image='Assets/boton_amarillo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.33, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Boton_Rojo = visual.ImageStim(
        win=win,
        name='Boton_Rojo', 
        image='Assets/boton_rojo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.54, -0.30), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Conteo_1 = visual.ImageStim(
        win=win,
        name='Conteo_1', 
        image='Assets/circulo_verde.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.32), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    Botones = visual.ImageStim(
        win=win,
        name='Botones', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.1), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    Cuenta_tarea1 = visual.TextStim(win=win, name='Cuenta_tarea1',
        text='',
        font='Arial',
        pos=(0, -0.32), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    # Run 'Begin Experiment' code from code
    contador = 1; 
    # set audio backend
    sound.Sound.backend = 'ptb'
    feedback = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='feedback'
    )
    feedback.setVolume(1.0)
    
    # --- Initialize components for Routine "Black_Screen" ---
    Wait = visual.Rect(
        win=win, name='Wait',
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Start_2" ---
    Arriba_Start = visual.ImageStim(
        win=win,
        name='Arriba_Start', 
        image='Assets/arriba.png', mask=None, anchor='center',
        ori=0.0, pos=(0.33, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Abajo_Start = visual.ImageStim(
        win=win,
        name='Abajo_Start', 
        image='Assets/abajo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.54, -0.30), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Indicaciones_4 = visual.TextStim(win=win, name='Indicaciones_4',
        text='¡Bienvenid@ al experimento!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
        font='Arial',
        pos=(0, 0.36), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    play_4 = visual.ImageStim(
        win=win,
        name='play_4', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "Instrucciones_2" ---
    Indicaciones_3 = visual.TextStim(win=win, name='Indicaciones_3',
        text='• Tienes 2 segundos para mover la palanca en la dirección que indica la flecha.\n\n• Solo puedes realizar movimientos hacia arriba, abajo, derecha e izquierda.\n\n• Después de cada movimiento, regresa la palanca a la posición inicial (Puedes mover las palancas para entender el movimiento antes de empezar).\n\nPresiona el botón verde para comenzar.',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    play_3 = visual.ImageStim(
        win=win,
        name='play_3', 
        image='Assets/Inicio.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.20), draggable=False, size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    Arriba_Start_2 = visual.ImageStim(
        win=win,
        name='Arriba_Start_2', 
        image='Assets/arriba.png', mask=None, anchor='center',
        ori=0.0, pos=(0.33, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    Abajo_Start_2 = visual.ImageStim(
        win=win,
        name='Abajo_Start_2', 
        image='Assets/abajo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.54, -0.30), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "Levers_Trial" ---
    Conteo = visual.ImageStim(
        win=win,
        name='Conteo', 
        image='Assets/circulo_verde.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.32), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Izquierda_2 = visual.ImageStim(
        win=win,
        name='Izquierda_2', 
        image='Assets/izquierda.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, -0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Derecha = visual.ImageStim(
        win=win,
        name='Derecha', 
        image='Assets/derecha.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    Arriba = visual.ImageStim(
        win=win,
        name='Arriba', 
        image='Assets/arriba.png', mask=None, anchor='center',
        ori=0.0, pos=(0.4, -0.1), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    Abajo = visual.ImageStim(
        win=win,
        name='Abajo', 
        image='Assets/abajo.png', mask=None, anchor='center',
        ori=0.0, pos=(0.4, -0.3), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    Instruccion = visual.ImageStim(
        win=win,
        name='Instruccion', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    Cuenta_tarea2 = visual.TextStim(win=win, name='Cuenta_tarea2',
        text='',
        font='Arial',
        pos=(0, -0.32), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    # Run 'Begin Experiment' code from code_4
    contador1 = 20; 
    feedback_2 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='feedback_2'
    )
    feedback_2.setVolume(1.0)
    
    # --- Initialize components for Routine "Black_Screen" ---
    Wait = visual.Rect(
        win=win, name='Wait',
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Start_Exp" ---
    # create an object to store info about Routine Start_Exp
    Start_Exp = data.Routine(
        name='Start_Exp',
        components=[Indicaciones_5, mouse_5, play_5],
    )
    Start_Exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_5
    mouse_5.x = []
    mouse_5.y = []
    mouse_5.leftButton = []
    mouse_5.midButton = []
    mouse_5.rightButton = []
    mouse_5.time = []
    mouse_5.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Start_Exp
    Start_Exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_Exp.tStart = globalClock.getTime(format='float')
    Start_Exp.status = STARTED
    thisExp.addData('Start_Exp.started', Start_Exp.tStart)
    Start_Exp.maxDuration = None
    # keep track of which components have finished
    Start_ExpComponents = Start_Exp.components
    for thisComponent in Start_Exp.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_Exp" ---
    thisExp.currentRoutine = Start_Exp
    Start_Exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Indicaciones_5* updates
        
        # if Indicaciones_5 is starting this frame...
        if Indicaciones_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones_5.frameNStart = frameN  # exact frame index
            Indicaciones_5.tStart = t  # local t and not account for scr refresh
            Indicaciones_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones_5.started')
            # update status
            Indicaciones_5.status = STARTED
            Indicaciones_5.setAutoDraw(True)
        
        # if Indicaciones_5 is active this frame...
        if Indicaciones_5.status == STARTED:
            # update params
            pass
        # *mouse_5* updates
        
        # if mouse_5 is starting this frame...
        if mouse_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_5.frameNStart = frameN  # exact frame index
            mouse_5.tStart = t  # local t and not account for scr refresh
            mouse_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_5.started', t)
            # update status
            mouse_5.status = STARTED
            mouse_5.mouseClock.reset()
            prevButtonState = mouse_5.getPressed()  # if button is down already this ISN'T a new click
        if mouse_5.status == STARTED:  # only update if started and not finished!
            buttons = mouse_5.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_5):
                            gotValidClick = True
                            mouse_5.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_5.clicked_name.append(None)
                    x, y = mouse_5.getPos()
                    mouse_5.x.append(float(x))
                    mouse_5.y.append(float(y))
                    buttons = mouse_5.getPressed()
                    mouse_5.leftButton.append(buttons[0])
                    mouse_5.midButton.append(buttons[1])
                    mouse_5.rightButton.append(buttons[2])
                    mouse_5.time.append(mouse_5.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play_5* updates
        
        # if play_5 is starting this frame...
        if play_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play_5.frameNStart = frameN  # exact frame index
            play_5.tStart = t  # local t and not account for scr refresh
            play_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play_5.started')
            # update status
            play_5.status = STARTED
            play_5.setAutoDraw(True)
        
        # if play_5 is active this frame...
        if play_5.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start_Exp,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start_Exp.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start_Exp.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start_Exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_Exp" ---
    for thisComponent in Start_Exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_Exp
    Start_Exp.tStop = globalClock.getTime(format='float')
    Start_Exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_Exp.stopped', Start_Exp.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_5.x', mouse_5.x)
    thisExp.addData('mouse_5.y', mouse_5.y)
    thisExp.addData('mouse_5.leftButton', mouse_5.leftButton)
    thisExp.addData('mouse_5.midButton', mouse_5.midButton)
    thisExp.addData('mouse_5.rightButton', mouse_5.rightButton)
    thisExp.addData('mouse_5.time', mouse_5.time)
    thisExp.addData('mouse_5.clicked_name', mouse_5.clicked_name)
    thisExp.nextEntry()
    # the Routine "Start_Exp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instruc_Emotions" ---
    # create an object to store info about Routine Instruc_Emotions
    Instruc_Emotions = data.Routine(
        name='Instruc_Emotions',
        components=[Indicaciones_6, mouse_6, play_6],
    )
    Instruc_Emotions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_6
    mouse_6.x = []
    mouse_6.y = []
    mouse_6.leftButton = []
    mouse_6.midButton = []
    mouse_6.rightButton = []
    mouse_6.time = []
    mouse_6.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Instruc_Emotions
    Instruc_Emotions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruc_Emotions.tStart = globalClock.getTime(format='float')
    Instruc_Emotions.status = STARTED
    thisExp.addData('Instruc_Emotions.started', Instruc_Emotions.tStart)
    Instruc_Emotions.maxDuration = None
    # keep track of which components have finished
    Instruc_EmotionsComponents = Instruc_Emotions.components
    for thisComponent in Instruc_Emotions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruc_Emotions" ---
    thisExp.currentRoutine = Instruc_Emotions
    Instruc_Emotions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Indicaciones_6* updates
        
        # if Indicaciones_6 is starting this frame...
        if Indicaciones_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones_6.frameNStart = frameN  # exact frame index
            Indicaciones_6.tStart = t  # local t and not account for scr refresh
            Indicaciones_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones_6.started')
            # update status
            Indicaciones_6.status = STARTED
            Indicaciones_6.setAutoDraw(True)
        
        # if Indicaciones_6 is active this frame...
        if Indicaciones_6.status == STARTED:
            # update params
            pass
        # *mouse_6* updates
        
        # if mouse_6 is starting this frame...
        if mouse_6.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_6.frameNStart = frameN  # exact frame index
            mouse_6.tStart = t  # local t and not account for scr refresh
            mouse_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_6.started', t)
            # update status
            mouse_6.status = STARTED
            mouse_6.mouseClock.reset()
            prevButtonState = mouse_6.getPressed()  # if button is down already this ISN'T a new click
        if mouse_6.status == STARTED:  # only update if started and not finished!
            buttons = mouse_6.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_6):
                            gotValidClick = True
                            mouse_6.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_6.clicked_name.append(None)
                    x, y = mouse_6.getPos()
                    mouse_6.x.append(float(x))
                    mouse_6.y.append(float(y))
                    buttons = mouse_6.getPressed()
                    mouse_6.leftButton.append(buttons[0])
                    mouse_6.midButton.append(buttons[1])
                    mouse_6.rightButton.append(buttons[2])
                    mouse_6.time.append(mouse_6.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play_6* updates
        
        # if play_6 is starting this frame...
        if play_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play_6.frameNStart = frameN  # exact frame index
            play_6.tStart = t  # local t and not account for scr refresh
            play_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play_6.started')
            # update status
            play_6.status = STARTED
            play_6.setAutoDraw(True)
        
        # if play_6 is active this frame...
        if play_6.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instruc_Emotions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instruc_Emotions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instruc_Emotions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instruc_Emotions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruc_Emotions" ---
    for thisComponent in Instruc_Emotions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruc_Emotions
    Instruc_Emotions.tStop = globalClock.getTime(format='float')
    Instruc_Emotions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruc_Emotions.stopped', Instruc_Emotions.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_6.x', mouse_6.x)
    thisExp.addData('mouse_6.y', mouse_6.y)
    thisExp.addData('mouse_6.leftButton', mouse_6.leftButton)
    thisExp.addData('mouse_6.midButton', mouse_6.midButton)
    thisExp.addData('mouse_6.rightButton', mouse_6.rightButton)
    thisExp.addData('mouse_6.time', mouse_6.time)
    thisExp.addData('mouse_6.clicked_name', mouse_6.clicked_name)
    thisExp.nextEntry()
    # the Routine "Instruc_Emotions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop1 = data.TrialHandler2(
        name='loop1',
        nReps=2, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Trial_3.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(loop1)  # add the loop to the experiment
    thisLoop1 = loop1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop1.rgb)
    if thisLoop1 != None:
        for paramName in thisLoop1:
            globals()[paramName] = thisLoop1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop1 in loop1:
        loop1.status = STARTED
        if hasattr(thisLoop1, 'status'):
            thisLoop1.status = STARTED
        currentLoop = loop1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop1.rgb)
        if thisLoop1 != None:
            for paramName in thisLoop1:
                globals()[paramName] = thisLoop1[paramName]
        
        # --- Prepare to start Routine "Emotions_Trial" ---
        # create an object to store info about Routine Emotions_Trial
        Emotions_Trial = data.Routine(
            name='Emotions_Trial',
            components=[Instruccion_2, circulo, Cuenta_tarea3, respuesta_slider],
        )
        Emotions_Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Instruccion_2.setImage(Emociones)
        Cuenta_tarea3.setText(contador)
        respuesta_slider.reset()
        # Run 'Begin Routine' code from Contador
        contador = contador + 1;
        respuesta = ""
        respuesta_recibida = False
        esp32.reset_input_buffer()
        # store start times for Emotions_Trial
        Emotions_Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Emotions_Trial.tStart = globalClock.getTime(format='float')
        Emotions_Trial.status = STARTED
        thisExp.addData('Emotions_Trial.started', Emotions_Trial.tStart)
        Emotions_Trial.maxDuration = None
        # keep track of which components have finished
        Emotions_TrialComponents = Emotions_Trial.components
        for thisComponent in Emotions_Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Emotions_Trial" ---
        thisExp.currentRoutine = Emotions_Trial
        Emotions_Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisLoop1, 'status') and thisLoop1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Instruccion_2* updates
            
            # if Instruccion_2 is starting this frame...
            if Instruccion_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instruccion_2.frameNStart = frameN  # exact frame index
                Instruccion_2.tStart = t  # local t and not account for scr refresh
                Instruccion_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instruccion_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Instruccion_2.started')
                # update status
                Instruccion_2.status = STARTED
                Instruccion_2.setAutoDraw(True)
            
            # if Instruccion_2 is active this frame...
            if Instruccion_2.status == STARTED:
                # update params
                pass
            
            # if Instruccion_2 is stopping this frame...
            if Instruccion_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Instruccion_2.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Instruccion_2.tStop = t  # not accounting for scr refresh
                    Instruccion_2.tStopRefresh = tThisFlipGlobal  # on global time
                    Instruccion_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Instruccion_2.stopped')
                    # update status
                    Instruccion_2.status = FINISHED
                    Instruccion_2.setAutoDraw(False)
            
            # *circulo* updates
            
            # if circulo is starting this frame...
            if circulo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                circulo.frameNStart = frameN  # exact frame index
                circulo.tStart = t  # local t and not account for scr refresh
                circulo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circulo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circulo.started')
                # update status
                circulo.status = STARTED
                circulo.setAutoDraw(True)
            
            # if circulo is active this frame...
            if circulo.status == STARTED:
                # update params
                pass
            
            # if circulo is stopping this frame...
            if circulo.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circulo.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    circulo.tStop = t  # not accounting for scr refresh
                    circulo.tStopRefresh = tThisFlipGlobal  # on global time
                    circulo.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'circulo.stopped')
                    # update status
                    circulo.status = FINISHED
                    circulo.setAutoDraw(False)
            
            # *Cuenta_tarea3* updates
            
            # if Cuenta_tarea3 is starting this frame...
            if Cuenta_tarea3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Cuenta_tarea3.frameNStart = frameN  # exact frame index
                Cuenta_tarea3.tStart = t  # local t and not account for scr refresh
                Cuenta_tarea3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Cuenta_tarea3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Cuenta_tarea3.started')
                # update status
                Cuenta_tarea3.status = STARTED
                Cuenta_tarea3.setAutoDraw(True)
            
            # if Cuenta_tarea3 is active this frame...
            if Cuenta_tarea3.status == STARTED:
                # update params
                pass
            
            # if Cuenta_tarea3 is stopping this frame...
            if Cuenta_tarea3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Cuenta_tarea3.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    Cuenta_tarea3.tStop = t  # not accounting for scr refresh
                    Cuenta_tarea3.tStopRefresh = tThisFlipGlobal  # on global time
                    Cuenta_tarea3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Cuenta_tarea3.stopped')
                    # update status
                    Cuenta_tarea3.status = FINISHED
                    Cuenta_tarea3.setAutoDraw(False)
            
            # *respuesta_slider* updates
            
            # if respuesta_slider is starting this frame...
            if respuesta_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                respuesta_slider.frameNStart = frameN  # exact frame index
                respuesta_slider.tStart = t  # local t and not account for scr refresh
                respuesta_slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(respuesta_slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'respuesta_slider.started')
                # update status
                respuesta_slider.status = STARTED
                respuesta_slider.setAutoDraw(True)
            
            # if respuesta_slider is active this frame...
            if respuesta_slider.status == STARTED:
                # update params
                pass
            
            # if respuesta_slider is stopping this frame...
            if respuesta_slider.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > respuesta_slider.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    respuesta_slider.tStop = t  # not accounting for scr refresh
                    respuesta_slider.tStopRefresh = tThisFlipGlobal  # on global time
                    respuesta_slider.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'respuesta_slider.stopped')
                    # update status
                    respuesta_slider.status = FINISHED
                    respuesta_slider.setAutoDraw(False)
            
            # Check respuesta_slider for response to end Routine
            if respuesta_slider.getRating() is not None and respuesta_slider.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Emotions_Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Emotions_Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Emotions_Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Emotions_Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Emotions_Trial" ---
        for thisComponent in Emotions_Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Emotions_Trial
        Emotions_Trial.tStop = globalClock.getTime(format='float')
        Emotions_Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Emotions_Trial.stopped', Emotions_Trial.tStop)
        loop1.addData('respuesta_slider.response', respuesta_slider.getRating())
        loop1.addData('respuesta_slider.rt', respuesta_slider.getRT())
        # Run 'End Routine' code from Contador
        thisExp.addData('respuesta', respuesta)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Emotions_Trial.maxDurationReached:
            routineTimer.addTime(-Emotions_Trial.maxDuration)
        elif Emotions_Trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "Black_Screen" ---
        # create an object to store info about Routine Black_Screen
        Black_Screen = data.Routine(
            name='Black_Screen',
            components=[Wait],
        )
        Black_Screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        isi_duration = random.uniform(1.0, 3.5)
        # store start times for Black_Screen
        Black_Screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Black_Screen.tStart = globalClock.getTime(format='float')
        Black_Screen.status = STARTED
        thisExp.addData('Black_Screen.started', Black_Screen.tStart)
        Black_Screen.maxDuration = None
        # keep track of which components have finished
        Black_ScreenComponents = Black_Screen.components
        for thisComponent in Black_Screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Black_Screen" ---
        thisExp.currentRoutine = Black_Screen
        Black_Screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLoop1, 'status') and thisLoop1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Wait* updates
            
            # if Wait is starting this frame...
            if Wait.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Wait.frameNStart = frameN  # exact frame index
                Wait.tStart = t  # local t and not account for scr refresh
                Wait.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Wait, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Wait.started')
                # update status
                Wait.status = STARTED
                Wait.setAutoDraw(True)
            
            # if Wait is active this frame...
            if Wait.status == STARTED:
                # update params
                pass
            
            # if Wait is stopping this frame...
            if Wait.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Wait.tStartRefresh + isi_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Wait.tStop = t  # not accounting for scr refresh
                    Wait.tStopRefresh = tThisFlipGlobal  # on global time
                    Wait.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Wait.stopped')
                    # update status
                    Wait.status = FINISHED
                    Wait.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
                
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Black_Screen,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Black_Screen.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Black_Screen.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Black_Screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Black_Screen" ---
        for thisComponent in Black_Screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Black_Screen
        Black_Screen.tStop = globalClock.getTime(format='float')
        Black_Screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Black_Screen.stopped', Black_Screen.tStop)
        # the Routine "Black_Screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisLoop1 as finished
        if hasattr(thisLoop1, 'status'):
            thisLoop1.status = FINISHED
        # if awaiting a pause, pause now
        if loop1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            loop1.status = STARTED
        thisExp.nextEntry()
        
    # completed 2 repeats of 'loop1'
    loop1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start" ---
    # create an object to store info about Routine Start
    Start = data.Routine(
        name='Start',
        components=[Boton_Amarillo_Start, Boton_Rojo_Start, Indicaciones, mouse_2, play],
    )
    Start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    mouse_2.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Start
    Start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start.tStart = globalClock.getTime(format='float')
    Start.status = STARTED
    thisExp.addData('Start.started', Start.tStart)
    Start.maxDuration = None
    # keep track of which components have finished
    StartComponents = Start.components
    for thisComponent in Start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    thisExp.currentRoutine = Start
    Start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Boton_Amarillo_Start* updates
        
        # if Boton_Amarillo_Start is starting this frame...
        if Boton_Amarillo_Start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Boton_Amarillo_Start.frameNStart = frameN  # exact frame index
            Boton_Amarillo_Start.tStart = t  # local t and not account for scr refresh
            Boton_Amarillo_Start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Boton_Amarillo_Start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Boton_Amarillo_Start.started')
            # update status
            Boton_Amarillo_Start.status = STARTED
            Boton_Amarillo_Start.setAutoDraw(True)
        
        # if Boton_Amarillo_Start is active this frame...
        if Boton_Amarillo_Start.status == STARTED:
            # update params
            pass
        
        # *Boton_Rojo_Start* updates
        
        # if Boton_Rojo_Start is starting this frame...
        if Boton_Rojo_Start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Boton_Rojo_Start.frameNStart = frameN  # exact frame index
            Boton_Rojo_Start.tStart = t  # local t and not account for scr refresh
            Boton_Rojo_Start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Boton_Rojo_Start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Boton_Rojo_Start.started')
            # update status
            Boton_Rojo_Start.status = STARTED
            Boton_Rojo_Start.setAutoDraw(True)
        
        # if Boton_Rojo_Start is active this frame...
        if Boton_Rojo_Start.status == STARTED:
            # update params
            pass
        
        # *Indicaciones* updates
        
        # if Indicaciones is starting this frame...
        if Indicaciones.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones.frameNStart = frameN  # exact frame index
            Indicaciones.tStart = t  # local t and not account for scr refresh
            Indicaciones.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones.started')
            # update status
            Indicaciones.status = STARTED
            Indicaciones.setAutoDraw(True)
        
        # if Indicaciones is active this frame...
        if Indicaciones.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_2.started', t)
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_2):
                            gotValidClick = True
                            mouse_2.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_2.clicked_name.append(None)
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(float(x))
                    mouse_2.y.append(float(y))
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play* updates
        
        # if play is starting this frame...
        if play.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play.frameNStart = frameN  # exact frame index
            play.tStart = t  # local t and not account for scr refresh
            play.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play.started')
            # update status
            play.status = STARTED
            play.setAutoDraw(True)
        
        # if play is active this frame...
        if play.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in Start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start
    Start.tStop = globalClock.getTime(format='float')
    Start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start.stopped', Start.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.addData('mouse_2.clicked_name', mouse_2.clicked_name)
    thisExp.nextEntry()
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instruc_Buttons" ---
    # create an object to store info about Routine Instruc_Buttons
    Instruc_Buttons = data.Routine(
        name='Instruc_Buttons',
        components=[Boton_Amarillo_Start_2, Boton_Rojo_Start_2, Indicaciones_2, mouse_4, play_2],
    )
    Instruc_Buttons.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_4
    mouse_4.x = []
    mouse_4.y = []
    mouse_4.leftButton = []
    mouse_4.midButton = []
    mouse_4.rightButton = []
    mouse_4.time = []
    mouse_4.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Instruc_Buttons
    Instruc_Buttons.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruc_Buttons.tStart = globalClock.getTime(format='float')
    Instruc_Buttons.status = STARTED
    thisExp.addData('Instruc_Buttons.started', Instruc_Buttons.tStart)
    Instruc_Buttons.maxDuration = None
    # keep track of which components have finished
    Instruc_ButtonsComponents = Instruc_Buttons.components
    for thisComponent in Instruc_Buttons.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instruc_Buttons" ---
    thisExp.currentRoutine = Instruc_Buttons
    Instruc_Buttons.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Boton_Amarillo_Start_2* updates
        
        # if Boton_Amarillo_Start_2 is starting this frame...
        if Boton_Amarillo_Start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Boton_Amarillo_Start_2.frameNStart = frameN  # exact frame index
            Boton_Amarillo_Start_2.tStart = t  # local t and not account for scr refresh
            Boton_Amarillo_Start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Boton_Amarillo_Start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Boton_Amarillo_Start_2.started')
            # update status
            Boton_Amarillo_Start_2.status = STARTED
            Boton_Amarillo_Start_2.setAutoDraw(True)
        
        # if Boton_Amarillo_Start_2 is active this frame...
        if Boton_Amarillo_Start_2.status == STARTED:
            # update params
            pass
        
        # *Boton_Rojo_Start_2* updates
        
        # if Boton_Rojo_Start_2 is starting this frame...
        if Boton_Rojo_Start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Boton_Rojo_Start_2.frameNStart = frameN  # exact frame index
            Boton_Rojo_Start_2.tStart = t  # local t and not account for scr refresh
            Boton_Rojo_Start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Boton_Rojo_Start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Boton_Rojo_Start_2.started')
            # update status
            Boton_Rojo_Start_2.status = STARTED
            Boton_Rojo_Start_2.setAutoDraw(True)
        
        # if Boton_Rojo_Start_2 is active this frame...
        if Boton_Rojo_Start_2.status == STARTED:
            # update params
            pass
        
        # *Indicaciones_2* updates
        
        # if Indicaciones_2 is starting this frame...
        if Indicaciones_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones_2.frameNStart = frameN  # exact frame index
            Indicaciones_2.tStart = t  # local t and not account for scr refresh
            Indicaciones_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones_2.started')
            # update status
            Indicaciones_2.status = STARTED
            Indicaciones_2.setAutoDraw(True)
        
        # if Indicaciones_2 is active this frame...
        if Indicaciones_2.status == STARTED:
            # update params
            pass
        # *mouse_4* updates
        
        # if mouse_4 is starting this frame...
        if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_4.frameNStart = frameN  # exact frame index
            mouse_4.tStart = t  # local t and not account for scr refresh
            mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_4.started', t)
            # update status
            mouse_4.status = STARTED
            mouse_4.mouseClock.reset()
            prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
        if mouse_4.status == STARTED:  # only update if started and not finished!
            buttons = mouse_4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_4):
                            gotValidClick = True
                            mouse_4.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_4.clicked_name.append(None)
                    x, y = mouse_4.getPos()
                    mouse_4.x.append(float(x))
                    mouse_4.y.append(float(y))
                    buttons = mouse_4.getPressed()
                    mouse_4.leftButton.append(buttons[0])
                    mouse_4.midButton.append(buttons[1])
                    mouse_4.rightButton.append(buttons[2])
                    mouse_4.time.append(mouse_4.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play_2* updates
        
        # if play_2 is starting this frame...
        if play_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play_2.frameNStart = frameN  # exact frame index
            play_2.tStart = t  # local t and not account for scr refresh
            play_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play_2.started')
            # update status
            play_2.status = STARTED
            play_2.setAutoDraw(True)
        
        # if play_2 is active this frame...
        if play_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instruc_Buttons,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instruc_Buttons.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instruc_Buttons.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instruc_Buttons.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruc_Buttons" ---
    for thisComponent in Instruc_Buttons.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruc_Buttons
    Instruc_Buttons.tStop = globalClock.getTime(format='float')
    Instruc_Buttons.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruc_Buttons.stopped', Instruc_Buttons.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_4.x', mouse_4.x)
    thisExp.addData('mouse_4.y', mouse_4.y)
    thisExp.addData('mouse_4.leftButton', mouse_4.leftButton)
    thisExp.addData('mouse_4.midButton', mouse_4.midButton)
    thisExp.addData('mouse_4.rightButton', mouse_4.rightButton)
    thisExp.addData('mouse_4.time', mouse_4.time)
    thisExp.addData('mouse_4.clicked_name', mouse_4.clicked_name)
    thisExp.nextEntry()
    # the Routine "Instruc_Buttons" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop2 = data.TrialHandler2(
        name='loop2',
        nReps=2, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Trial_1.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(loop2)  # add the loop to the experiment
    thisLoop2 = loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop2.rgb)
    if thisLoop2 != None:
        for paramName in thisLoop2:
            globals()[paramName] = thisLoop2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop2 in loop2:
        loop2.status = STARTED
        if hasattr(thisLoop2, 'status'):
            thisLoop2.status = STARTED
        currentLoop = loop2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop2.rgb)
        if thisLoop2 != None:
            for paramName in thisLoop2:
                globals()[paramName] = thisLoop2[paramName]
        
        # --- Prepare to start Routine "Buttons_Trial" ---
        # create an object to store info about Routine Buttons_Trial
        Buttons_Trial = data.Routine(
            name='Buttons_Trial',
            components=[Boton_Amarillo, Boton_Rojo, Conteo_1, Botones, Cuenta_tarea1, feedback],
        )
        Buttons_Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Botones.setImage(Botones_Random)
        Cuenta_tarea1.setText(contador)
        # Run 'Begin Routine' code from code
        contador = contador + 1
        
        respuesta = ""
        respuesta_recibida = False
        marker_enviado = False
        
        feedbackPlayed = False
        feedbackStart = None
        
        esp32.reset_input_buffer()
        feedback.setSound('incorrect.mp3', secs=1.0, hamming=True)
        feedback.setVolume(1.0, log=False)
        feedback.seek(0)
        # Run 'Begin Routine' code from Timestamps2
        marker_enviado = False
        
        # store start times for Buttons_Trial
        Buttons_Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Buttons_Trial.tStart = globalClock.getTime(format='float')
        Buttons_Trial.status = STARTED
        thisExp.addData('Buttons_Trial.started', Buttons_Trial.tStart)
        Buttons_Trial.maxDuration = None
        # keep track of which components have finished
        Buttons_TrialComponents = Buttons_Trial.components
        for thisComponent in Buttons_Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Buttons_Trial" ---
        thisExp.currentRoutine = Buttons_Trial
        Buttons_Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLoop2, 'status') and thisLoop2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Boton_Amarillo* updates
            
            # if Boton_Amarillo is starting this frame...
            if Boton_Amarillo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Boton_Amarillo.frameNStart = frameN  # exact frame index
                Boton_Amarillo.tStart = t  # local t and not account for scr refresh
                Boton_Amarillo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Boton_Amarillo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Boton_Amarillo.started')
                # update status
                Boton_Amarillo.status = STARTED
                Boton_Amarillo.setAutoDraw(True)
            
            # if Boton_Amarillo is active this frame...
            if Boton_Amarillo.status == STARTED:
                # update params
                pass
            
            # *Boton_Rojo* updates
            
            # if Boton_Rojo is starting this frame...
            if Boton_Rojo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Boton_Rojo.frameNStart = frameN  # exact frame index
                Boton_Rojo.tStart = t  # local t and not account for scr refresh
                Boton_Rojo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Boton_Rojo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Boton_Rojo.started')
                # update status
                Boton_Rojo.status = STARTED
                Boton_Rojo.setAutoDraw(True)
            
            # if Boton_Rojo is active this frame...
            if Boton_Rojo.status == STARTED:
                # update params
                pass
            
            # *Conteo_1* updates
            
            # if Conteo_1 is starting this frame...
            if Conteo_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Conteo_1.frameNStart = frameN  # exact frame index
                Conteo_1.tStart = t  # local t and not account for scr refresh
                Conteo_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Conteo_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Conteo_1.started')
                # update status
                Conteo_1.status = STARTED
                Conteo_1.setAutoDraw(True)
            
            # if Conteo_1 is active this frame...
            if Conteo_1.status == STARTED:
                # update params
                pass
            
            # *Botones* updates
            
            # if Botones is starting this frame...
            if Botones.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Botones.frameNStart = frameN  # exact frame index
                Botones.tStart = t  # local t and not account for scr refresh
                Botones.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Botones, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Botones.started')
                # update status
                Botones.status = STARTED
                Botones.setAutoDraw(True)
            
            # if Botones is active this frame...
            if Botones.status == STARTED:
                # update params
                pass
            
            # *Cuenta_tarea1* updates
            
            # if Cuenta_tarea1 is starting this frame...
            if Cuenta_tarea1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Cuenta_tarea1.frameNStart = frameN  # exact frame index
                Cuenta_tarea1.tStart = t  # local t and not account for scr refresh
                Cuenta_tarea1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Cuenta_tarea1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Cuenta_tarea1.started')
                # update status
                Cuenta_tarea1.status = STARTED
                Cuenta_tarea1.setAutoDraw(True)
            
            # if Cuenta_tarea1 is active this frame...
            if Cuenta_tarea1.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code
            if not respuesta_recibida:
            
                if esp32.in_waiting > 0:
            
                    try:
                        dato = esp32.readline().decode(errors='ignore').strip()
            
                        print(dato)
            
                        if dato in ["AMARILLO", "ROJO", "EMPTY"]:
                            marker_outlet.push(MARKERS.RESP_BUTTON)
                            respuesta = dato
                            respuesta_recibida = True
                            feedbackStart = t
              
            
                    except Exception as e:
                        print(e)
            
            
            # Si no respondieron y se acabó el tiempo
            if t >= 1.3 and not respuesta_recibida:
            
                respuesta = 'NONE'
                respuesta_recibida = True
            
                feedbackStart = t
            
            
            # Reproducir feedback UNA sola vez
            if respuesta_recibida and not feedbackPlayed:
            
                if respuesta == Correct:
                    feedback.setSound('correct.mp3')
                else:
                    feedback.setSound('incorrect.mp3')
                feedback.play()
            
                feedbackPlayed = True
            
            
            # Esperar 0.5 s y terminar rutina
            if feedbackPlayed and t >= feedbackStart + 0.5:
            
                continueRoutine = False
                
                
            
            # *feedback* updates
            
            # if feedback is starting this frame...
            if feedback.status == NOT_STARTED and False:
                # keep track of start time/frame for later
                feedback.frameNStart = frameN  # exact frame index
                feedback.tStart = t  # local t and not account for scr refresh
                feedback.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('feedback.started', t)
                # update status
                feedback.status = STARTED
                feedback.play()  # start the sound (it finishes automatically)
            
            # if feedback is stopping this frame...
            if feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback.tStartRefresh + 1.0-frameTolerance or feedback.isFinished:
                    # keep track of stop time/frame for later
                    feedback.tStop = t  # not accounting for scr refresh
                    feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('feedback.stopped', t)
                    # update status
                    feedback.status = FINISHED
                    feedback.stop()
            # Run 'Each Frame' code from Timestamps2
            if Botones.status == STARTED and not marker_enviado:
            
                nombre = os.path.basename(Botones_Random)
            
                if nombre in ['amarillo.jpg', 'rojo.jpg']:
                    win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
                else:
                    win.callOnFlip(marker_outlet.push, MARKERS.STIM_NOGO)
            
                marker_enviado = True
                
            if respuesta_recibida and not feedbackPlayed:
            
                nombre = os.path.basename(Botones_Random)
                es_go = nombre in ['amarillo.jpg', 'rojo.jpg']
            
                # Marcadores FTI
                if not es_go:
            
                    if respuesta == 'NONE':
                        marker_outlet.push(MARKERS.CORRECT_INHIBITION)
            
                    elif feedbackStart < 0.200:
                        marker_outlet.push(MARKERS.FTI_BALLISTIC_ERROR)
            
                    else:
                        marker_outlet.push(MARKERS.CONTROLLED_RESPONSE)
            
                # Feedback auditivo
                if respuesta == Correct:
                    feedback.setSound('correct.mp3')
            
                else:
                    feedback.setSound('incorrect.mp3')
            
                feedback.play()
                feedbackPlayed = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Buttons_Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Buttons_Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Buttons_Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Buttons_Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Buttons_Trial" ---
        for thisComponent in Buttons_Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Buttons_Trial
        Buttons_Trial.tStop = globalClock.getTime(format='float')
        Buttons_Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Buttons_Trial.stopped', Buttons_Trial.tStop)
        # Run 'End Routine' code from code
        thisExp.addData('respuesta', respuesta)
        feedback.pause()  # ensure sound has stopped at end of Routine
        # the Routine "Buttons_Trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Black_Screen" ---
        # create an object to store info about Routine Black_Screen
        Black_Screen = data.Routine(
            name='Black_Screen',
            components=[Wait],
        )
        Black_Screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        isi_duration = random.uniform(1.0, 3.5)
        # store start times for Black_Screen
        Black_Screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Black_Screen.tStart = globalClock.getTime(format='float')
        Black_Screen.status = STARTED
        thisExp.addData('Black_Screen.started', Black_Screen.tStart)
        Black_Screen.maxDuration = None
        # keep track of which components have finished
        Black_ScreenComponents = Black_Screen.components
        for thisComponent in Black_Screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Black_Screen" ---
        thisExp.currentRoutine = Black_Screen
        Black_Screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLoop2, 'status') and thisLoop2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Wait* updates
            
            # if Wait is starting this frame...
            if Wait.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Wait.frameNStart = frameN  # exact frame index
                Wait.tStart = t  # local t and not account for scr refresh
                Wait.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Wait, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Wait.started')
                # update status
                Wait.status = STARTED
                Wait.setAutoDraw(True)
            
            # if Wait is active this frame...
            if Wait.status == STARTED:
                # update params
                pass
            
            # if Wait is stopping this frame...
            if Wait.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Wait.tStartRefresh + isi_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Wait.tStop = t  # not accounting for scr refresh
                    Wait.tStopRefresh = tThisFlipGlobal  # on global time
                    Wait.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Wait.stopped')
                    # update status
                    Wait.status = FINISHED
                    Wait.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
                
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Black_Screen,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Black_Screen.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Black_Screen.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Black_Screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Black_Screen" ---
        for thisComponent in Black_Screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Black_Screen
        Black_Screen.tStop = globalClock.getTime(format='float')
        Black_Screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Black_Screen.stopped', Black_Screen.tStop)
        # the Routine "Black_Screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisLoop2 as finished
        if hasattr(thisLoop2, 'status'):
            thisLoop2.status = FINISHED
        # if awaiting a pause, pause now
        if loop2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            loop2.status = STARTED
        thisExp.nextEntry()
        
    # completed 2 repeats of 'loop2'
    loop2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start_2" ---
    # create an object to store info about Routine Start_2
    Start_2 = data.Routine(
        name='Start_2',
        components=[Arriba_Start, Abajo_Start, Indicaciones_4, mouse_3, play_4],
    )
    Start_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_3
    mouse_3.x = []
    mouse_3.y = []
    mouse_3.leftButton = []
    mouse_3.midButton = []
    mouse_3.rightButton = []
    mouse_3.time = []
    mouse_3.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Start_2
    Start_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_2.tStart = globalClock.getTime(format='float')
    Start_2.status = STARTED
    thisExp.addData('Start_2.started', Start_2.tStart)
    Start_2.maxDuration = None
    # keep track of which components have finished
    Start_2Components = Start_2.components
    for thisComponent in Start_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_2" ---
    thisExp.currentRoutine = Start_2
    Start_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Arriba_Start* updates
        
        # if Arriba_Start is starting this frame...
        if Arriba_Start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Arriba_Start.frameNStart = frameN  # exact frame index
            Arriba_Start.tStart = t  # local t and not account for scr refresh
            Arriba_Start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Arriba_Start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Arriba_Start.started')
            # update status
            Arriba_Start.status = STARTED
            Arriba_Start.setAutoDraw(True)
        
        # if Arriba_Start is active this frame...
        if Arriba_Start.status == STARTED:
            # update params
            pass
        
        # *Abajo_Start* updates
        
        # if Abajo_Start is starting this frame...
        if Abajo_Start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Abajo_Start.frameNStart = frameN  # exact frame index
            Abajo_Start.tStart = t  # local t and not account for scr refresh
            Abajo_Start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Abajo_Start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Abajo_Start.started')
            # update status
            Abajo_Start.status = STARTED
            Abajo_Start.setAutoDraw(True)
        
        # if Abajo_Start is active this frame...
        if Abajo_Start.status == STARTED:
            # update params
            pass
        
        # *Indicaciones_4* updates
        
        # if Indicaciones_4 is starting this frame...
        if Indicaciones_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones_4.frameNStart = frameN  # exact frame index
            Indicaciones_4.tStart = t  # local t and not account for scr refresh
            Indicaciones_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones_4.started')
            # update status
            Indicaciones_4.status = STARTED
            Indicaciones_4.setAutoDraw(True)
        
        # if Indicaciones_4 is active this frame...
        if Indicaciones_4.status == STARTED:
            # update params
            pass
        # *mouse_3* updates
        
        # if mouse_3 is starting this frame...
        if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_3.frameNStart = frameN  # exact frame index
            mouse_3.tStart = t  # local t and not account for scr refresh
            mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_3.started', t)
            # update status
            mouse_3.status = STARTED
            mouse_3.mouseClock.reset()
            prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
        if mouse_3.status == STARTED:  # only update if started and not finished!
            buttons = mouse_3.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_3):
                            gotValidClick = True
                            mouse_3.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse_3.clicked_name.append(None)
                    x, y = mouse_3.getPos()
                    mouse_3.x.append(float(x))
                    mouse_3.y.append(float(y))
                    buttons = mouse_3.getPressed()
                    mouse_3.leftButton.append(buttons[0])
                    mouse_3.midButton.append(buttons[1])
                    mouse_3.rightButton.append(buttons[2])
                    mouse_3.time.append(mouse_3.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play_4* updates
        
        # if play_4 is starting this frame...
        if play_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play_4.frameNStart = frameN  # exact frame index
            play_4.tStart = t  # local t and not account for scr refresh
            play_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play_4.started')
            # update status
            play_4.status = STARTED
            play_4.setAutoDraw(True)
        
        # if play_4 is active this frame...
        if play_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Start_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Start_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Start_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_2" ---
    for thisComponent in Start_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_2
    Start_2.tStop = globalClock.getTime(format='float')
    Start_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_2.stopped', Start_2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_3.x', mouse_3.x)
    thisExp.addData('mouse_3.y', mouse_3.y)
    thisExp.addData('mouse_3.leftButton', mouse_3.leftButton)
    thisExp.addData('mouse_3.midButton', mouse_3.midButton)
    thisExp.addData('mouse_3.rightButton', mouse_3.rightButton)
    thisExp.addData('mouse_3.time', mouse_3.time)
    thisExp.addData('mouse_3.clicked_name', mouse_3.clicked_name)
    thisExp.nextEntry()
    # the Routine "Start_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instrucciones_2" ---
    # create an object to store info about Routine Instrucciones_2
    Instrucciones_2 = data.Routine(
        name='Instrucciones_2',
        components=[Indicaciones_3, mouse, play_3, Arriba_Start_2, Abajo_Start_2],
    )
    Instrucciones_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # store start times for Instrucciones_2
    Instrucciones_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instrucciones_2.tStart = globalClock.getTime(format='float')
    Instrucciones_2.status = STARTED
    thisExp.addData('Instrucciones_2.started', Instrucciones_2.tStart)
    Instrucciones_2.maxDuration = None
    # keep track of which components have finished
    Instrucciones_2Components = Instrucciones_2.components
    for thisComponent in Instrucciones_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instrucciones_2" ---
    thisExp.currentRoutine = Instrucciones_2
    Instrucciones_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Indicaciones_3* updates
        
        # if Indicaciones_3 is starting this frame...
        if Indicaciones_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Indicaciones_3.frameNStart = frameN  # exact frame index
            Indicaciones_3.tStart = t  # local t and not account for scr refresh
            Indicaciones_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Indicaciones_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Indicaciones_3.started')
            # update status
            Indicaciones_3.status = STARTED
            Indicaciones_3.setAutoDraw(True)
        
        # if Indicaciones_3 is active this frame...
        if Indicaciones_3.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse.started', t)
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(play_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse):
                            gotValidClick = True
                            mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        mouse.clicked_name.append(None)
                    x, y = mouse.getPos()
                    mouse.x.append(float(x))
                    mouse.y.append(float(y))
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *play_3* updates
        
        # if play_3 is starting this frame...
        if play_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            play_3.frameNStart = frameN  # exact frame index
            play_3.tStart = t  # local t and not account for scr refresh
            play_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(play_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'play_3.started')
            # update status
            play_3.status = STARTED
            play_3.setAutoDraw(True)
        
        # if play_3 is active this frame...
        if play_3.status == STARTED:
            # update params
            pass
        
        # *Arriba_Start_2* updates
        
        # if Arriba_Start_2 is starting this frame...
        if Arriba_Start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Arriba_Start_2.frameNStart = frameN  # exact frame index
            Arriba_Start_2.tStart = t  # local t and not account for scr refresh
            Arriba_Start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Arriba_Start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Arriba_Start_2.started')
            # update status
            Arriba_Start_2.status = STARTED
            Arriba_Start_2.setAutoDraw(True)
        
        # if Arriba_Start_2 is active this frame...
        if Arriba_Start_2.status == STARTED:
            # update params
            pass
        
        # *Abajo_Start_2* updates
        
        # if Abajo_Start_2 is starting this frame...
        if Abajo_Start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Abajo_Start_2.frameNStart = frameN  # exact frame index
            Abajo_Start_2.tStart = t  # local t and not account for scr refresh
            Abajo_Start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Abajo_Start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Abajo_Start_2.started')
            # update status
            Abajo_Start_2.status = STARTED
            Abajo_Start_2.setAutoDraw(True)
        
        # if Abajo_Start_2 is active this frame...
        if Abajo_Start_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instrucciones_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instrucciones_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instrucciones_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instrucciones_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instrucciones_2" ---
    for thisComponent in Instrucciones_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instrucciones_2
    Instrucciones_2.tStop = globalClock.getTime(format='float')
    Instrucciones_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instrucciones_2.stopped', Instrucciones_2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.addData('mouse.clicked_name', mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "Instrucciones_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop3 = data.TrialHandler2(
        name='loop3',
        nReps=3, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Trial_2.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(loop3)  # add the loop to the experiment
    thisLoop3 = loop3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop3.rgb)
    if thisLoop3 != None:
        for paramName in thisLoop3:
            globals()[paramName] = thisLoop3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop3 in loop3:
        loop3.status = STARTED
        if hasattr(thisLoop3, 'status'):
            thisLoop3.status = STARTED
        currentLoop = loop3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop3.rgb)
        if thisLoop3 != None:
            for paramName in thisLoop3:
                globals()[paramName] = thisLoop3[paramName]
        
        # --- Prepare to start Routine "Levers_Trial" ---
        # create an object to store info about Routine Levers_Trial
        Levers_Trial = data.Routine(
            name='Levers_Trial',
            components=[Conteo, Izquierda_2, Derecha, Arriba, Abajo, Instruccion, Cuenta_tarea2, feedback_2],
        )
        Levers_Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Instruccion.setImage(Flechas_Random)
        Cuenta_tarea2.setText(contador1)
        # Run 'Begin Routine' code from code_4
        contador1 = contador1 + 1;
        respuesta = ""
        respuesta_recibida = False
        feedbackPlayed = False
        feedbackStart = None
        esp32_1.reset_input_buffer()
        feedback_2.setSound('incorrect.mp3', secs=1.0, hamming=True)
        feedback_2.setVolume(1.0, log=False)
        feedback_2.seek(0)
        # Run 'Begin Routine' code from Timestamps_3
        marker_enviado = False
        # store start times for Levers_Trial
        Levers_Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Levers_Trial.tStart = globalClock.getTime(format='float')
        Levers_Trial.status = STARTED
        thisExp.addData('Levers_Trial.started', Levers_Trial.tStart)
        Levers_Trial.maxDuration = None
        # keep track of which components have finished
        Levers_TrialComponents = Levers_Trial.components
        for thisComponent in Levers_Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Levers_Trial" ---
        thisExp.currentRoutine = Levers_Trial
        Levers_Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLoop3, 'status') and thisLoop3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Conteo* updates
            
            # if Conteo is starting this frame...
            if Conteo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Conteo.frameNStart = frameN  # exact frame index
                Conteo.tStart = t  # local t and not account for scr refresh
                Conteo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Conteo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Conteo.started')
                # update status
                Conteo.status = STARTED
                Conteo.setAutoDraw(True)
            
            # if Conteo is active this frame...
            if Conteo.status == STARTED:
                # update params
                pass
            
            # *Izquierda_2* updates
            
            # if Izquierda_2 is starting this frame...
            if Izquierda_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Izquierda_2.frameNStart = frameN  # exact frame index
                Izquierda_2.tStart = t  # local t and not account for scr refresh
                Izquierda_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Izquierda_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Izquierda_2.started')
                # update status
                Izquierda_2.status = STARTED
                Izquierda_2.setAutoDraw(True)
            
            # if Izquierda_2 is active this frame...
            if Izquierda_2.status == STARTED:
                # update params
                pass
            
            # *Derecha* updates
            
            # if Derecha is starting this frame...
            if Derecha.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Derecha.frameNStart = frameN  # exact frame index
                Derecha.tStart = t  # local t and not account for scr refresh
                Derecha.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Derecha, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Derecha.started')
                # update status
                Derecha.status = STARTED
                Derecha.setAutoDraw(True)
            
            # if Derecha is active this frame...
            if Derecha.status == STARTED:
                # update params
                pass
            
            # *Arriba* updates
            
            # if Arriba is starting this frame...
            if Arriba.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Arriba.frameNStart = frameN  # exact frame index
                Arriba.tStart = t  # local t and not account for scr refresh
                Arriba.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Arriba, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Arriba.started')
                # update status
                Arriba.status = STARTED
                Arriba.setAutoDraw(True)
            
            # if Arriba is active this frame...
            if Arriba.status == STARTED:
                # update params
                pass
            
            # *Abajo* updates
            
            # if Abajo is starting this frame...
            if Abajo.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Abajo.frameNStart = frameN  # exact frame index
                Abajo.tStart = t  # local t and not account for scr refresh
                Abajo.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Abajo, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Abajo.started')
                # update status
                Abajo.status = STARTED
                Abajo.setAutoDraw(True)
            
            # if Abajo is active this frame...
            if Abajo.status == STARTED:
                # update params
                pass
            
            # *Instruccion* updates
            
            # if Instruccion is starting this frame...
            if Instruccion.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instruccion.frameNStart = frameN  # exact frame index
                Instruccion.tStart = t  # local t and not account for scr refresh
                Instruccion.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instruccion, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Instruccion.started')
                # update status
                Instruccion.status = STARTED
                Instruccion.setAutoDraw(True)
            
            # if Instruccion is active this frame...
            if Instruccion.status == STARTED:
                # update params
                pass
            
            # *Cuenta_tarea2* updates
            
            # if Cuenta_tarea2 is starting this frame...
            if Cuenta_tarea2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Cuenta_tarea2.frameNStart = frameN  # exact frame index
                Cuenta_tarea2.tStart = t  # local t and not account for scr refresh
                Cuenta_tarea2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Cuenta_tarea2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Cuenta_tarea2.started')
                # update status
                Cuenta_tarea2.status = STARTED
                Cuenta_tarea2.setAutoDraw(True)
            
            # if Cuenta_tarea2 is active this frame...
            if Cuenta_tarea2.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_4
            if not respuesta_recibida:
            
                if esp32_1.in_waiting > 0:
            
                    try:
            
                        dato = esp32_1.readline().decode(errors='ignore').strip()
            
                        print(dato)
            
                        if dato in ["ARRIBA", "ABAJO", "IZQUIERDA", "DERECHA"]:
            
                            respuesta = dato
            
                            respuesta_recibida = True
                            
                            feedbackStart = t
                            
                            if dato in ["ARRIBA", "ABAJO"]:
                                marker_outlet.push(MARKERS.RESP_LEVER_R)
            
                            elif dato in ["IZQUIERDA", "DERECHA"]:
                                marker_outlet.push(MARKERS.RESP_LEVER_L)
            
                    except:
                        pass
                        
            if t >= 1.6 and not respuesta_recibida:
            
                respuesta = 'NONE'
                respuesta_recibida = True
            
                feedbackStart = t
                
            if respuesta_recibida and not feedbackPlayed:
            
                if respuesta == Correct:
                    feedback_2.setSound('correct.mp3')
                else:
                    feedback_2.setSound('incorrect.mp3')
            
                feedback_2.play()
            
                feedbackPlayed = True
            
            
            # Esperar 0.5 s y terminar rutina
            if feedbackPlayed and t >= feedbackStart + 0.5:
            
                continueRoutine = False
            
            # *feedback_2* updates
            
            # if feedback_2 is starting this frame...
            if feedback_2.status == NOT_STARTED and False:
                # keep track of start time/frame for later
                feedback_2.frameNStart = frameN  # exact frame index
                feedback_2.tStart = t  # local t and not account for scr refresh
                feedback_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('feedback_2.started', t)
                # update status
                feedback_2.status = STARTED
                feedback_2.play()  # start the sound (it finishes automatically)
            
            # if feedback_2 is stopping this frame...
            if feedback_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_2.tStartRefresh + 1.0-frameTolerance or feedback_2.isFinished:
                    # keep track of stop time/frame for later
                    feedback_2.tStop = t  # not accounting for scr refresh
                    feedback_2.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('feedback_2.stopped', t)
                    # update status
                    feedback_2.status = FINISHED
                    feedback_2.stop()
            # Run 'Each Frame' code from Timestamps_3
            if Instruccion.status == STARTED and not marker_enviado:
                win.callOnFlip(marker_outlet.push, MARKERS.STIM_GO)
                marker_enviado = True
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Levers_Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Levers_Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Levers_Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Levers_Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Levers_Trial" ---
        for thisComponent in Levers_Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Levers_Trial
        Levers_Trial.tStop = globalClock.getTime(format='float')
        Levers_Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Levers_Trial.stopped', Levers_Trial.tStop)
        # Run 'End Routine' code from code_4
        thisExp.addData('respuesta', respuesta)
        feedback_2.pause()  # ensure sound has stopped at end of Routine
        # the Routine "Levers_Trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Black_Screen" ---
        # create an object to store info about Routine Black_Screen
        Black_Screen = data.Routine(
            name='Black_Screen',
            components=[Wait],
        )
        Black_Screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        isi_duration = random.uniform(1.0, 3.5)
        # store start times for Black_Screen
        Black_Screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Black_Screen.tStart = globalClock.getTime(format='float')
        Black_Screen.status = STARTED
        thisExp.addData('Black_Screen.started', Black_Screen.tStart)
        Black_Screen.maxDuration = None
        # keep track of which components have finished
        Black_ScreenComponents = Black_Screen.components
        for thisComponent in Black_Screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Black_Screen" ---
        thisExp.currentRoutine = Black_Screen
        Black_Screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLoop3, 'status') and thisLoop3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Wait* updates
            
            # if Wait is starting this frame...
            if Wait.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Wait.frameNStart = frameN  # exact frame index
                Wait.tStart = t  # local t and not account for scr refresh
                Wait.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Wait, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Wait.started')
                # update status
                Wait.status = STARTED
                Wait.setAutoDraw(True)
            
            # if Wait is active this frame...
            if Wait.status == STARTED:
                # update params
                pass
            
            # if Wait is stopping this frame...
            if Wait.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Wait.tStartRefresh + isi_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Wait.tStop = t  # not accounting for scr refresh
                    Wait.tStopRefresh = tThisFlipGlobal  # on global time
                    Wait.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Wait.stopped')
                    # update status
                    Wait.status = FINISHED
                    Wait.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
                
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Black_Screen,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Black_Screen.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Black_Screen.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Black_Screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Black_Screen" ---
        for thisComponent in Black_Screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Black_Screen
        Black_Screen.tStop = globalClock.getTime(format='float')
        Black_Screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Black_Screen.stopped', Black_Screen.tStop)
        # the Routine "Black_Screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisLoop3 as finished
        if hasattr(thisLoop3, 'status'):
            thisLoop3.status = FINISHED
        # if awaiting a pause, pause now
        if loop3.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            loop3.status = STARTED
        thisExp.nextEntry()
        
    # completed 3 repeats of 'loop3'
    loop3.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # Run 'End Experiment' code from Serial_Begin
    esp32_1.close()
    esp32.close()
    # Run 'End Experiment' code from EEG_Start_Code
    marker_outlet.push(MARKERS.BLOCK_END)
    print("EEG marker sent: BLOCK_END")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
