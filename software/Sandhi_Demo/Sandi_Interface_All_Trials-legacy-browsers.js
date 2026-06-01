/************************ 
 * Sandi_Interface *
 ************************/


// store info about the experiment session:
let expName = 'Sandi_Interface';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};
let PILOTING = util.getUrlParameters().has('__pilotToken');

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(Start_3RoutineBegin());
flowScheduler.add(Start_3RoutineEachFrame());
flowScheduler.add(Start_3RoutineEnd());
flowScheduler.add(Instrucciones_3RoutineBegin());
flowScheduler.add(Instrucciones_3RoutineEachFrame());
flowScheduler.add(Instrucciones_3RoutineEnd());
const trials_3LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_3LoopBegin(trials_3LoopScheduler));
flowScheduler.add(trials_3LoopScheduler);
flowScheduler.add(trials_3LoopEnd);


flowScheduler.add(StartRoutineBegin());
flowScheduler.add(StartRoutineEachFrame());
flowScheduler.add(StartRoutineEnd());
flowScheduler.add(InstruccionesRoutineBegin());
flowScheduler.add(InstruccionesRoutineEachFrame());
flowScheduler.add(InstruccionesRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);


flowScheduler.add(Start_2RoutineBegin());
flowScheduler.add(Start_2RoutineEachFrame());
flowScheduler.add(Start_2RoutineEnd());
flowScheduler.add(Instrucciones_2RoutineBegin());
flowScheduler.add(Instrucciones_2RoutineEachFrame());
flowScheduler.add(Instrucciones_2RoutineEnd());
const trials_2LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_2LoopBegin(trials_2LoopScheduler));
flowScheduler.add(trials_2LoopScheduler);
flowScheduler.add(trials_2LoopEnd);


flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'Trial_3.csv', 'path': 'Trial_3.csv'},
    {'name': 'Assets/Felicidad.jpg', 'path': 'Assets/Felicidad.jpg'},
    {'name': 'Assets/Sorpresa.jpg', 'path': 'Assets/Sorpresa.jpg'},
    {'name': 'Assets/Desagrado.jpg', 'path': 'Assets/Desagrado.jpg'},
    {'name': 'Assets/Tristeza.jpg', 'path': 'Assets/Tristeza.jpg'},
    {'name': 'Trial_1.csv', 'path': 'Trial_1.csv'},
    {'name': 'Assets/boton_rojo.png', 'path': 'Assets/boton_rojo.png'},
    {'name': 'Assets/boton_amarillo.png', 'path': 'Assets/boton_amarillo.png'},
    {'name': 'Assets/boton_azul.png', 'path': 'Assets/boton_azul.png'},
    {'name': 'Assets/boton_verde.png', 'path': 'Assets/boton_verde.png'},
    {'name': 'Trial_2.csv', 'path': 'Trial_2.csv'},
    {'name': 'Assets/arriba.png', 'path': 'Assets/arriba.png'},
    {'name': 'Assets/abajo.png', 'path': 'Assets/abajo.png'},
    {'name': 'Assets/derecha.png', 'path': 'Assets/derecha.png'},
    {'name': 'Assets/izquierda.png', 'path': 'Assets/izquierda.png'},
    {'name': 'Assets/Inicio.png', 'path': 'Assets/Inicio.png'},
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
    {'name': 'Assets/boton_amarillo.png', 'path': 'Assets/boton_amarillo.png'},
    {'name': 'Assets/boton_rojo.png', 'path': 'Assets/boton_rojo.png'},
    {'name': 'Assets/circulo_verde.png', 'path': 'Assets/circulo_verde.png'},
    {'name': 'incorrect.mp3', 'path': 'incorrect.mp3'},
    {'name': 'Assets/arriba.png', 'path': 'Assets/arriba.png'},
    {'name': 'Assets/abajo.png', 'path': 'Assets/abajo.png'},
    {'name': 'Assets/izquierda.png', 'path': 'Assets/izquierda.png'},
    {'name': 'Assets/derecha.png', 'path': 'Assets/derecha.png'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);

async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2026.1.3';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}

async function experimentInit() {
  // Initialize components for Routine "Start_3"
  Start_3Clock = new util.Clock();
  Indicaciones_5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones_5',
    text: '¡Bienvenid@ a Sandhi Interface!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.36], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  mouse_5 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_5.mouseClock = new util.Clock();
  play_5 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play_5', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "Instrucciones_3"
  Instrucciones_3Clock = new util.Clock();
  Indicaciones_6 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones_6',
    text: '• Tienes 3 segundos para identificar la emoción mostrada en la imagen.\n\n• Utiliza el slider para registrar tu respuesta.\n\nPresiona el botón verde para comenzar.\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.36], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  mouse_6 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_6.mouseClock = new util.Clock();
  play_6 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play_6', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, (- 0.2)], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "trial_3"
  trial_3Clock = new util.Clock();
  Instruccion_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Instruccion_2', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0.2], 
    draggable: false,
    size : [0.4, 0.4],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Cuenta_tarea3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Cuenta_tarea3',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.32)], draggable: false, height: 0.08,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Run 'Begin Experiment' code from code_6
  contador = 1;
  
  slider = new visual.Slider({
    win: psychoJS.window, name: 'slider',
    startValue: undefined,
    size: [1.0, 0.1], pos: [0, (- 0.15)], ori: 0.0, units: psychoJS.window.units,
    labels: ["Alegr\u00eda", "Sorpresa", "Desagrado", "Tristeza"], fontSize: 0.04, ticks: [1, 2, 3, 4],
    granularity: 0.0, style: ["RATING"],
    color: new util.Color((-1.0000, -1.0000, -1.0000)), markerColor: new util.Color('Red'), lineColor: new util.Color('White'), 
    opacity: undefined, fontFamily: 'Noto Sans', bold: true, italic: false, depth: -3, 
    flip: false,
  });
  
  // Initialize components for Routine "Start"
  StartClock = new util.Clock();
  Boton_Amarillo_Start = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Amarillo_Start', units : undefined, 
    image : 'Assets/boton_amarillo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.33, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Boton_Rojo_Start = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Rojo_Start', units : undefined, 
    image : 'Assets/boton_rojo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.54, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  Indicaciones = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones',
    text: '¡Bienvenid@ a Sandhi Interface!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.36], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  mouse_2 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_2.mouseClock = new util.Clock();
  play = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  // Run 'Begin Experiment' code from code_2
  import * as serial from 'serial';
  esp32 = new serial.Serial("COM3", 115200, {"timeout": 0.01});
  
  // Initialize components for Routine "Instrucciones"
  InstruccionesClock = new util.Clock();
  Boton_Amarillo_Start_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Amarillo_Start_2', units : undefined, 
    image : 'Assets/boton_amarillo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.33, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Boton_Rojo_Start_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Rojo_Start_2', units : undefined, 
    image : 'Assets/boton_rojo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.54, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  Indicaciones_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones_2',
    text: '• En la botonera presiona el botón del color que aparece en pantalla.\n\n• Si sale un color distinto al amarillo o al rojo, no deberás presionar ningún botón.\n\n•Tienes 1.5s para seleccionar el botón\n\nPresiona el botón verde para empezar',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.25], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  mouse_4 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_4.mouseClock = new util.Clock();
  play_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play_2', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, (- 0.2)], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  Boton_Amarillo = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Amarillo', units : undefined, 
    image : 'Assets/boton_amarillo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.33, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Boton_Rojo = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Boton_Rojo', units : undefined, 
    image : 'Assets/boton_rojo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.54, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  Conteo_1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Conteo_1', units : undefined, 
    image : 'Assets/circulo_verde.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, (- 0.32)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  Botones = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Botones', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0.1], 
    draggable: false,
    size : [0.6, 0.6],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  Cuenta_tarea1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Cuenta_tarea1',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.32)], draggable: false, height: 0.08,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  // Run 'Begin Experiment' code from code
  contador = 1;
  
  feedback = new sound.Sound({
      win: psychoJS.window,
      value: 'A',
      secs: 1.0,
      });
  feedback.setVolume(1.0);
  feedback.isPlaying = false;
  feedback.isFinished = false;
  // Initialize components for Routine "Start_2"
  Start_2Clock = new util.Clock();
  Arriba_Start = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Arriba_Start', units : undefined, 
    image : 'Assets/arriba.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.33, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Abajo_Start = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Abajo_Start', units : undefined, 
    image : 'Assets/abajo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.54, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  Indicaciones_4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones_4',
    text: '¡Bienvenid@ al experimento!\nSigue las indicaciones que te aparecerán en pantalla.\n¡Presiona el botón verde para iniciar!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.36], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  mouse_3 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_3.mouseClock = new util.Clock();
  play_4 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play_4', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  // Initialize components for Routine "Instrucciones_2"
  Instrucciones_2Clock = new util.Clock();
  Indicaciones_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Indicaciones_3',
    text: '• Tienes 2 segundos para mover la palanca en la dirección que indica la flecha.\n\n• Solo puedes realizar movimientos hacia arriba, abajo, derecha e izquierda.\n\n• Después de cada movimiento, regresa la palanca a la posición inicial (Puedes mover las palancas para entender el movimiento antes de empezar).\n\nPresiona el botón verde para comenzar.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0.2], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  mouse = new core.Mouse({
    win: psychoJS.window,
  });
  mouse.mouseClock = new util.Clock();
  play_3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'play_3', units : undefined, 
    image : 'Assets/Inicio.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, (- 0.2)], 
    draggable: false,
    size : [0.3, 0.3],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  Arriba_Start_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Arriba_Start_2', units : undefined, 
    image : 'Assets/arriba.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.33, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  Abajo_Start_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Abajo_Start_2', units : undefined, 
    image : 'Assets/abajo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.54, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  // Initialize components for Routine "Trial_2"
  Trial_2Clock = new util.Clock();
  Izquierda_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Izquierda_2', units : undefined, 
    image : 'Assets/izquierda.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.5), (- 0.2)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  Derecha = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Derecha', units : undefined, 
    image : 'Assets/derecha.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.3), (- 0.2)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  Arriba = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Arriba', units : undefined, 
    image : 'Assets/arriba.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.4, (- 0.1)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  Abajo = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Abajo', units : undefined, 
    image : 'Assets/abajo.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.4, (- 0.3)], 
    draggable: false,
    size : [0.2, 0.2],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  Instruccion = new visual.ImageStim({
    win : psychoJS.window,
    name : 'Instruccion', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0.2], 
    draggable: false,
    size : [0.4, 0.4],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  Cuenta_tarea2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'Cuenta_tarea2',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.32)], draggable: false, height: 0.08,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -5.0 
  });
  
  // Run 'Begin Experiment' code from code_4
  contador = 1;
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

function Start_3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Start_3' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    Start_3Clock.reset();
    routineTimer.reset();
    Start_3MaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_5
    // current position of the mouse:
    mouse_5.x = [];
    mouse_5.y = [];
    mouse_5.leftButton = [];
    mouse_5.midButton = [];
    mouse_5.rightButton = [];
    mouse_5.time = [];
    mouse_5.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Start_3.started', globalClock.getTime());
    Start_3MaxDuration = null
    // keep track of which components have finished
    Start_3Components = [];
    Start_3Components.push(Indicaciones_5);
    Start_3Components.push(mouse_5);
    Start_3Components.push(play_5);
    
    Start_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function Start_3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Start_3' ---
    // get current time
    t = Start_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Indicaciones_5* updates
    if (t >= 0.0 && Indicaciones_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones_5.tStart = t;  // (not accounting for frame time here)
      Indicaciones_5.frameNStart = frameN;  // exact frame index
      
      Indicaciones_5.setAutoDraw(true);
    }
    
    
    // if Indicaciones_5 is active this frame...
    if (Indicaciones_5.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse_5* updates
    if (t >= 0.0 && mouse_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_5.tStart = t;  // (not accounting for frame time here)
      mouse_5.frameNStart = frameN;  // exact frame index
      
      mouse_5.status = PsychoJS.Status.STARTED;
      mouse_5.mouseClock.reset();
      prevButtonState = mouse_5.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse_5 is active this frame...
    if (mouse_5.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse_5.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_5.clickableObjects = eval(play)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_5.clickableObjects)) {
              mouse_5.clickableObjects = [mouse_5.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_5.clickableObjects) {
              if (obj.contains(mouse_5)) {
                  gotValidClick = true;
                  mouse_5.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_5.clicked_name.push(null);
          }
          _mouseXYs = mouse_5.getPos();
          mouse_5.x.push(_mouseXYs[0]);
          mouse_5.y.push(_mouseXYs[1]);
          mouse_5.leftButton.push(_mouseButtons[0]);
          mouse_5.midButton.push(_mouseButtons[1]);
          mouse_5.rightButton.push(_mouseButtons[2]);
          mouse_5.time.push(mouse_5.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play_5* updates
    if (t >= 0.0 && play_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play_5.tStart = t;  // (not accounting for frame time here)
      play_5.frameNStart = frameN;  // exact frame index
      
      play_5.setAutoDraw(true);
    }
    
    
    // if play_5 is active this frame...
    if (play_5.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Start_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function Start_3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Start_3' ---
    Start_3Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Start_3.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_5.x', mouse_5.x);
    psychoJS.experiment.addData('mouse_5.y', mouse_5.y);
    psychoJS.experiment.addData('mouse_5.leftButton', mouse_5.leftButton);
    psychoJS.experiment.addData('mouse_5.midButton', mouse_5.midButton);
    psychoJS.experiment.addData('mouse_5.rightButton', mouse_5.rightButton);
    psychoJS.experiment.addData('mouse_5.time', mouse_5.time);
    psychoJS.experiment.addData('mouse_5.clicked_name', mouse_5.clicked_name);
    
    // the Routine "Start_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function Instrucciones_3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Instrucciones_3' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    Instrucciones_3Clock.reset();
    routineTimer.reset();
    Instrucciones_3MaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_6
    // current position of the mouse:
    mouse_6.x = [];
    mouse_6.y = [];
    mouse_6.leftButton = [];
    mouse_6.midButton = [];
    mouse_6.rightButton = [];
    mouse_6.time = [];
    mouse_6.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Instrucciones_3.started', globalClock.getTime());
    Instrucciones_3MaxDuration = null
    // keep track of which components have finished
    Instrucciones_3Components = [];
    Instrucciones_3Components.push(Indicaciones_6);
    Instrucciones_3Components.push(mouse_6);
    Instrucciones_3Components.push(play_6);
    
    Instrucciones_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function Instrucciones_3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Instrucciones_3' ---
    // get current time
    t = Instrucciones_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Indicaciones_6* updates
    if (t >= 0.0 && Indicaciones_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones_6.tStart = t;  // (not accounting for frame time here)
      Indicaciones_6.frameNStart = frameN;  // exact frame index
      
      Indicaciones_6.setAutoDraw(true);
    }
    
    
    // if Indicaciones_6 is active this frame...
    if (Indicaciones_6.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse_6* updates
    if (t >= 0.0 && mouse_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_6.tStart = t;  // (not accounting for frame time here)
      mouse_6.frameNStart = frameN;  // exact frame index
      
      mouse_6.status = PsychoJS.Status.STARTED;
      mouse_6.mouseClock.reset();
      prevButtonState = mouse_6.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse_6 is active this frame...
    if (mouse_6.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse_6.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_6.clickableObjects = eval(play_2)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_6.clickableObjects)) {
              mouse_6.clickableObjects = [mouse_6.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_6.clickableObjects) {
              if (obj.contains(mouse_6)) {
                  gotValidClick = true;
                  mouse_6.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_6.clicked_name.push(null);
          }
          _mouseXYs = mouse_6.getPos();
          mouse_6.x.push(_mouseXYs[0]);
          mouse_6.y.push(_mouseXYs[1]);
          mouse_6.leftButton.push(_mouseButtons[0]);
          mouse_6.midButton.push(_mouseButtons[1]);
          mouse_6.rightButton.push(_mouseButtons[2]);
          mouse_6.time.push(mouse_6.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play_6* updates
    if (t >= 0.0 && play_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play_6.tStart = t;  // (not accounting for frame time here)
      play_6.frameNStart = frameN;  // exact frame index
      
      play_6.setAutoDraw(true);
    }
    
    
    // if play_6 is active this frame...
    if (play_6.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Instrucciones_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function Instrucciones_3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Instrucciones_3' ---
    Instrucciones_3Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Instrucciones_3.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_6.x', mouse_6.x);
    psychoJS.experiment.addData('mouse_6.y', mouse_6.y);
    psychoJS.experiment.addData('mouse_6.leftButton', mouse_6.leftButton);
    psychoJS.experiment.addData('mouse_6.midButton', mouse_6.midButton);
    psychoJS.experiment.addData('mouse_6.rightButton', mouse_6.rightButton);
    psychoJS.experiment.addData('mouse_6.time', mouse_6.time);
    psychoJS.experiment.addData('mouse_6.clicked_name', mouse_6.clicked_name);
    
    // the Routine "Instrucciones_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function trials_3LoopBegin(trials_3LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials_3 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'Trial_3.csv',
      seed: undefined, name: 'trials_3'
    });
    psychoJS.experiment.addLoop(trials_3); // add the loop to the experiment
    currentLoop = trials_3;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials_3.forEach(function() {
      snapshot = trials_3.getSnapshot();
    
      trials_3LoopScheduler.add(importConditions(snapshot));
      trials_3LoopScheduler.add(trial_3RoutineBegin(snapshot));
      trials_3LoopScheduler.add(trial_3RoutineEachFrame());
      trials_3LoopScheduler.add(trial_3RoutineEnd(snapshot));
      trials_3LoopScheduler.add(trials_3LoopEndIteration(trials_3LoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}

async function trials_3LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials_3);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trials_3LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 2, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'Trial_1.csv',
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(trialRoutineBegin(snapshot));
      trialsLoopScheduler.add(trialRoutineEachFrame());
      trialsLoopScheduler.add(trialRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}

async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function trials_2LoopBegin(trials_2LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials_2 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 3, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'Trial_2.csv',
      seed: undefined, name: 'trials_2'
    });
    psychoJS.experiment.addLoop(trials_2); // add the loop to the experiment
    currentLoop = trials_2;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials_2.forEach(function() {
      snapshot = trials_2.getSnapshot();
    
      trials_2LoopScheduler.add(importConditions(snapshot));
      trials_2LoopScheduler.add(Trial_2RoutineBegin(snapshot));
      trials_2LoopScheduler.add(Trial_2RoutineEachFrame());
      trials_2LoopScheduler.add(Trial_2RoutineEnd(snapshot));
      trials_2LoopScheduler.add(trials_2LoopEndIteration(trials_2LoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}

async function trials_2LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials_2);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trials_2LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function trial_3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial_3' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    trial_3Clock.reset(routineTimer.getTime());
    routineTimer.add(3.000000);
    trial_3MaxDurationReached = false;
    // update component parameters for each repeat
    Instruccion_2.setImage(Emociones);
    Cuenta_tarea3.setText(contador);
    // Run 'Begin Routine' code from code_6
    contador = (contador + 1);
    respuesta = "";
    respuesta_recibida = false;
    esp32.reset_input_buffer();
    
    slider.reset()
    psychoJS.experiment.addData('trial_3.started', globalClock.getTime());
    trial_3MaxDuration = null
    // keep track of which components have finished
    trial_3Components = [];
    trial_3Components.push(Instruccion_2);
    trial_3Components.push(Cuenta_tarea3);
    trial_3Components.push(slider);
    
    trial_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function trial_3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial_3' ---
    // get current time
    t = trial_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Instruccion_2* updates
    if (t >= 0.0 && Instruccion_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Instruccion_2.tStart = t;  // (not accounting for frame time here)
      Instruccion_2.frameNStart = frameN;  // exact frame index
      
      Instruccion_2.setAutoDraw(true);
    }
    
    
    // if Instruccion_2 is active this frame...
    if (Instruccion_2.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (Instruccion_2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      Instruccion_2.tStop = t;  // not accounting for scr refresh
      Instruccion_2.frameNStop = frameN;  // exact frame index
      // update status
      Instruccion_2.status = PsychoJS.Status.FINISHED;
      Instruccion_2.setAutoDraw(false);
    }
    
    
    // *Cuenta_tarea3* updates
    if (t >= 0.0 && Cuenta_tarea3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Cuenta_tarea3.tStart = t;  // (not accounting for frame time here)
      Cuenta_tarea3.frameNStart = frameN;  // exact frame index
      
      Cuenta_tarea3.setAutoDraw(true);
    }
    
    
    // if Cuenta_tarea3 is active this frame...
    if (Cuenta_tarea3.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (Cuenta_tarea3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      Cuenta_tarea3.tStop = t;  // not accounting for scr refresh
      Cuenta_tarea3.frameNStop = frameN;  // exact frame index
      // update status
      Cuenta_tarea3.status = PsychoJS.Status.FINISHED;
      Cuenta_tarea3.setAutoDraw(false);
    }
    
    // Run 'Each Frame' code from code_6
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if ((! respuesta_recibida)) {
        if ((esp32.in_waiting > 0)) {
            try {
                dato = esp32.readline().decode({"errors": "ignore"}).strip();
                console.log(dato);
                if (_pj.in_es6(dato, ["AMARILLO", "ROJO"])) {
                    respuesta = dato;
                    respuesta_recibida = true;
                    continueRoutine = false;
                }
            } catch(e) {
            }
        }
    }
    
    
    // *slider* updates
    if (t >= 0.0 && slider.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      slider.tStart = t;  // (not accounting for frame time here)
      slider.frameNStart = frameN;  // exact frame index
      
      slider.setAutoDraw(true);
    }
    
    
    // if slider is active this frame...
    if (slider.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 3.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (slider.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      slider.tStop = t;  // not accounting for scr refresh
      slider.frameNStop = frameN;  // exact frame index
      // update status
      slider.status = PsychoJS.Status.FINISHED;
      slider.setAutoDraw(false);
    }
    
    
    // Check slider for response to end Routine
    if (slider.getRating() !== undefined && slider.status === PsychoJS.Status.STARTED) {
      continueRoutine = false; }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trial_3Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function trial_3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial_3' ---
    trial_3Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trial_3.stopped', globalClock.getTime());
    // Run 'End Routine' code from code_6
    psychoJS.experiment.addData("respuesta", respuesta);
    
    psychoJS.experiment.addData('slider.response', slider.getRating());
    psychoJS.experiment.addData('slider.rt', slider.getRT());
    if (routineForceEnded) {
        routineTimer.reset();} else if (trial_3MaxDurationReached) {
        trial_3Clock.add(trial_3MaxDuration);
    } else {
        trial_3Clock.add(3.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function StartRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Start' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    StartClock.reset();
    routineTimer.reset();
    StartMaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_2
    // current position of the mouse:
    mouse_2.x = [];
    mouse_2.y = [];
    mouse_2.leftButton = [];
    mouse_2.midButton = [];
    mouse_2.rightButton = [];
    mouse_2.time = [];
    mouse_2.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Start.started', globalClock.getTime());
    StartMaxDuration = null
    // keep track of which components have finished
    StartComponents = [];
    StartComponents.push(Boton_Amarillo_Start);
    StartComponents.push(Boton_Rojo_Start);
    StartComponents.push(Indicaciones);
    StartComponents.push(mouse_2);
    StartComponents.push(play);
    
    StartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function StartRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Start' ---
    // get current time
    t = StartClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Boton_Amarillo_Start* updates
    if (t >= 0.0 && Boton_Amarillo_Start.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Amarillo_Start.tStart = t;  // (not accounting for frame time here)
      Boton_Amarillo_Start.frameNStart = frameN;  // exact frame index
      
      Boton_Amarillo_Start.setAutoDraw(true);
    }
    
    
    // if Boton_Amarillo_Start is active this frame...
    if (Boton_Amarillo_Start.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Boton_Rojo_Start* updates
    if (t >= 0.0 && Boton_Rojo_Start.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Rojo_Start.tStart = t;  // (not accounting for frame time here)
      Boton_Rojo_Start.frameNStart = frameN;  // exact frame index
      
      Boton_Rojo_Start.setAutoDraw(true);
    }
    
    
    // if Boton_Rojo_Start is active this frame...
    if (Boton_Rojo_Start.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Indicaciones* updates
    if (t >= 0.0 && Indicaciones.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones.tStart = t;  // (not accounting for frame time here)
      Indicaciones.frameNStart = frameN;  // exact frame index
      
      Indicaciones.setAutoDraw(true);
    }
    
    
    // if Indicaciones is active this frame...
    if (Indicaciones.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse_2* updates
    if (t >= 0.0 && mouse_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_2.tStart = t;  // (not accounting for frame time here)
      mouse_2.frameNStart = frameN;  // exact frame index
      
      mouse_2.status = PsychoJS.Status.STARTED;
      mouse_2.mouseClock.reset();
      prevButtonState = mouse_2.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse_2 is active this frame...
    if (mouse_2.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse_2.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_2.clickableObjects = eval(play)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_2.clickableObjects)) {
              mouse_2.clickableObjects = [mouse_2.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_2.clickableObjects) {
              if (obj.contains(mouse_2)) {
                  gotValidClick = true;
                  mouse_2.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_2.clicked_name.push(null);
          }
          _mouseXYs = mouse_2.getPos();
          mouse_2.x.push(_mouseXYs[0]);
          mouse_2.y.push(_mouseXYs[1]);
          mouse_2.leftButton.push(_mouseButtons[0]);
          mouse_2.midButton.push(_mouseButtons[1]);
          mouse_2.rightButton.push(_mouseButtons[2]);
          mouse_2.time.push(mouse_2.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play* updates
    if (t >= 0.0 && play.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play.tStart = t;  // (not accounting for frame time here)
      play.frameNStart = frameN;  // exact frame index
      
      play.setAutoDraw(true);
    }
    
    
    // if play is active this frame...
    if (play.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    StartComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function StartRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Start' ---
    StartComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Start.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_2.x', mouse_2.x);
    psychoJS.experiment.addData('mouse_2.y', mouse_2.y);
    psychoJS.experiment.addData('mouse_2.leftButton', mouse_2.leftButton);
    psychoJS.experiment.addData('mouse_2.midButton', mouse_2.midButton);
    psychoJS.experiment.addData('mouse_2.rightButton', mouse_2.rightButton);
    psychoJS.experiment.addData('mouse_2.time', mouse_2.time);
    psychoJS.experiment.addData('mouse_2.clicked_name', mouse_2.clicked_name);
    
    // the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function InstruccionesRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Instrucciones' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    InstruccionesClock.reset();
    routineTimer.reset();
    InstruccionesMaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_4
    // current position of the mouse:
    mouse_4.x = [];
    mouse_4.y = [];
    mouse_4.leftButton = [];
    mouse_4.midButton = [];
    mouse_4.rightButton = [];
    mouse_4.time = [];
    mouse_4.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Instrucciones.started', globalClock.getTime());
    InstruccionesMaxDuration = null
    // keep track of which components have finished
    InstruccionesComponents = [];
    InstruccionesComponents.push(Boton_Amarillo_Start_2);
    InstruccionesComponents.push(Boton_Rojo_Start_2);
    InstruccionesComponents.push(Indicaciones_2);
    InstruccionesComponents.push(mouse_4);
    InstruccionesComponents.push(play_2);
    
    InstruccionesComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function InstruccionesRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Instrucciones' ---
    // get current time
    t = InstruccionesClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Boton_Amarillo_Start_2* updates
    if (t >= 0.0 && Boton_Amarillo_Start_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Amarillo_Start_2.tStart = t;  // (not accounting for frame time here)
      Boton_Amarillo_Start_2.frameNStart = frameN;  // exact frame index
      
      Boton_Amarillo_Start_2.setAutoDraw(true);
    }
    
    
    // if Boton_Amarillo_Start_2 is active this frame...
    if (Boton_Amarillo_Start_2.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Boton_Rojo_Start_2* updates
    if (t >= 0.0 && Boton_Rojo_Start_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Rojo_Start_2.tStart = t;  // (not accounting for frame time here)
      Boton_Rojo_Start_2.frameNStart = frameN;  // exact frame index
      
      Boton_Rojo_Start_2.setAutoDraw(true);
    }
    
    
    // if Boton_Rojo_Start_2 is active this frame...
    if (Boton_Rojo_Start_2.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Indicaciones_2* updates
    if (t >= 0.0 && Indicaciones_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones_2.tStart = t;  // (not accounting for frame time here)
      Indicaciones_2.frameNStart = frameN;  // exact frame index
      
      Indicaciones_2.setAutoDraw(true);
    }
    
    
    // if Indicaciones_2 is active this frame...
    if (Indicaciones_2.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse_4* updates
    if (t >= 0.0 && mouse_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_4.tStart = t;  // (not accounting for frame time here)
      mouse_4.frameNStart = frameN;  // exact frame index
      
      mouse_4.status = PsychoJS.Status.STARTED;
      mouse_4.mouseClock.reset();
      prevButtonState = mouse_4.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse_4 is active this frame...
    if (mouse_4.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse_4.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_4.clickableObjects = eval(play_2)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_4.clickableObjects)) {
              mouse_4.clickableObjects = [mouse_4.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_4.clickableObjects) {
              if (obj.contains(mouse_4)) {
                  gotValidClick = true;
                  mouse_4.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_4.clicked_name.push(null);
          }
          _mouseXYs = mouse_4.getPos();
          mouse_4.x.push(_mouseXYs[0]);
          mouse_4.y.push(_mouseXYs[1]);
          mouse_4.leftButton.push(_mouseButtons[0]);
          mouse_4.midButton.push(_mouseButtons[1]);
          mouse_4.rightButton.push(_mouseButtons[2]);
          mouse_4.time.push(mouse_4.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play_2* updates
    if (t >= 0.0 && play_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play_2.tStart = t;  // (not accounting for frame time here)
      play_2.frameNStart = frameN;  // exact frame index
      
      play_2.setAutoDraw(true);
    }
    
    
    // if play_2 is active this frame...
    if (play_2.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    InstruccionesComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function InstruccionesRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Instrucciones' ---
    InstruccionesComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Instrucciones.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_4.x', mouse_4.x);
    psychoJS.experiment.addData('mouse_4.y', mouse_4.y);
    psychoJS.experiment.addData('mouse_4.leftButton', mouse_4.leftButton);
    psychoJS.experiment.addData('mouse_4.midButton', mouse_4.midButton);
    psychoJS.experiment.addData('mouse_4.rightButton', mouse_4.rightButton);
    psychoJS.experiment.addData('mouse_4.time', mouse_4.time);
    psychoJS.experiment.addData('mouse_4.clicked_name', mouse_4.clicked_name);
    
    // the Routine "Instrucciones" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    trialClock.reset();
    routineTimer.reset();
    trialMaxDurationReached = false;
    // update component parameters for each repeat
    Botones.setImage(Botones_Random);
    Cuenta_tarea1.setText(contador);
    // Run 'Begin Routine' code from code
    contador = (contador + 1);
    respuesta = "";
    respuesta_recibida = false;
    feedbackPlayed = false;
    feedbackStart = null;
    esp32.reset_input_buffer();
    
    feedback.isFinished = false;
    feedback.setValue('incorrect.mp3');
    feedback.secs=1.0;
    feedback.setVolume(1.0);
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    trialMaxDuration = null
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(Boton_Amarillo);
    trialComponents.push(Boton_Rojo);
    trialComponents.push(Conteo_1);
    trialComponents.push(Botones);
    trialComponents.push(Cuenta_tarea1);
    trialComponents.push(feedback);
    
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Boton_Amarillo* updates
    if (t >= 0 && Boton_Amarillo.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Amarillo.tStart = t;  // (not accounting for frame time here)
      Boton_Amarillo.frameNStart = frameN;  // exact frame index
      
      Boton_Amarillo.setAutoDraw(true);
    }
    
    
    // if Boton_Amarillo is active this frame...
    if (Boton_Amarillo.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Boton_Rojo* updates
    if (t >= 0 && Boton_Rojo.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Boton_Rojo.tStart = t;  // (not accounting for frame time here)
      Boton_Rojo.frameNStart = frameN;  // exact frame index
      
      Boton_Rojo.setAutoDraw(true);
    }
    
    
    // if Boton_Rojo is active this frame...
    if (Boton_Rojo.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Conteo_1* updates
    if (t >= 0 && Conteo_1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Conteo_1.tStart = t;  // (not accounting for frame time here)
      Conteo_1.frameNStart = frameN;  // exact frame index
      
      Conteo_1.setAutoDraw(true);
    }
    
    
    // if Conteo_1 is active this frame...
    if (Conteo_1.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Botones* updates
    if (t >= 0.0 && Botones.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Botones.tStart = t;  // (not accounting for frame time here)
      Botones.frameNStart = frameN;  // exact frame index
      
      Botones.setAutoDraw(true);
    }
    
    
    // if Botones is active this frame...
    if (Botones.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Cuenta_tarea1* updates
    if (t >= 0.0 && Cuenta_tarea1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Cuenta_tarea1.tStart = t;  // (not accounting for frame time here)
      Cuenta_tarea1.frameNStart = frameN;  // exact frame index
      
      Cuenta_tarea1.setAutoDraw(true);
    }
    
    
    // if Cuenta_tarea1 is active this frame...
    if (Cuenta_tarea1.status === PsychoJS.Status.STARTED) {
    }
    
    // Run 'Each Frame' code from code
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if ((! respuesta_recibida)) {
        if ((esp32.in_waiting > 0)) {
            try {
                dato = esp32.readline().decode({"errors": "ignore"}).strip();
                console.log(dato);
                if (_pj.in_es6(dato, ["AMARILLO", "ROJO", "EMPTY"])) {
                    respuesta = dato;
                    respuesta_recibida = true;
                    feedbackStart = t;
                }
            } catch(e) {
                console.log(e);
            }
        }
    }
    if (((t >= 1.3) && (! respuesta_recibida))) {
        respuesta = "NONE";
        respuesta_recibida = true;
        feedbackStart = t;
    }
    if ((respuesta_recibida && (! feedbackPlayed))) {
        if ((respuesta === Correct)) {
            feedback.setSound("correct.mp3");
        } else {
            feedback.setSound("incorrect.mp3");
        }
        feedback.play();
        feedbackPlayed = true;
    }
    if ((feedbackPlayed && (t >= (feedbackStart + 0.5)))) {
        continueRoutine = false;
    }
    
    if (feedback.status === STARTED) {
        feedback.isPlaying = true;
        if (t >= (feedback.getDuration() + feedback.tStart)) {
            feedback.isFinished = true;
        }
    }
    // start/stop feedback
    if ((false) && feedback.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      feedback.tStart = t;  // (not accounting for frame time here)
      feedback.frameNStart = frameN;  // exact frame index
      
      feedback.play();  // start the sound (it finishes automatically)
      feedback.status = PsychoJS.Status.STARTED;
    }
    if (feedback.status === PsychoJS.Status.STARTED && t >= (feedback.tStart + 1.0) || feedback.isFinished) {
      // keep track of stop time/frame for later
      feedback.tStop = t;  // not accounting for scr refresh
      feedback.frameNStop = frameN;  // exact frame index
      // update status
      feedback.status = PsychoJS.Status.FINISHED;
      // stop playback
      feedback.stop();
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function trialRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial' ---
    trialComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
    // Run 'End Routine' code from code
    psychoJS.experiment.addData("respuesta", respuesta);
    
    feedback.stop();  // ensure sound has stopped at end of Routine
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function Start_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Start_2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    Start_2Clock.reset();
    routineTimer.reset();
    Start_2MaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_3
    // current position of the mouse:
    mouse_3.x = [];
    mouse_3.y = [];
    mouse_3.leftButton = [];
    mouse_3.midButton = [];
    mouse_3.rightButton = [];
    mouse_3.time = [];
    mouse_3.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Start_2.started', globalClock.getTime());
    Start_2MaxDuration = null
    // keep track of which components have finished
    Start_2Components = [];
    Start_2Components.push(Arriba_Start);
    Start_2Components.push(Abajo_Start);
    Start_2Components.push(Indicaciones_4);
    Start_2Components.push(mouse_3);
    Start_2Components.push(play_4);
    
    Start_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function Start_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Start_2' ---
    // get current time
    t = Start_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Arriba_Start* updates
    if (t >= 0.0 && Arriba_Start.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Arriba_Start.tStart = t;  // (not accounting for frame time here)
      Arriba_Start.frameNStart = frameN;  // exact frame index
      
      Arriba_Start.setAutoDraw(true);
    }
    
    
    // if Arriba_Start is active this frame...
    if (Arriba_Start.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Abajo_Start* updates
    if (t >= 0.0 && Abajo_Start.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Abajo_Start.tStart = t;  // (not accounting for frame time here)
      Abajo_Start.frameNStart = frameN;  // exact frame index
      
      Abajo_Start.setAutoDraw(true);
    }
    
    
    // if Abajo_Start is active this frame...
    if (Abajo_Start.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Indicaciones_4* updates
    if (t >= 0.0 && Indicaciones_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones_4.tStart = t;  // (not accounting for frame time here)
      Indicaciones_4.frameNStart = frameN;  // exact frame index
      
      Indicaciones_4.setAutoDraw(true);
    }
    
    
    // if Indicaciones_4 is active this frame...
    if (Indicaciones_4.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse_3* updates
    if (t >= 0.0 && mouse_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_3.tStart = t;  // (not accounting for frame time here)
      mouse_3.frameNStart = frameN;  // exact frame index
      
      mouse_3.status = PsychoJS.Status.STARTED;
      mouse_3.mouseClock.reset();
      prevButtonState = mouse_3.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse_3 is active this frame...
    if (mouse_3.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse_3.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_3.clickableObjects = eval(play)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_3.clickableObjects)) {
              mouse_3.clickableObjects = [mouse_3.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_3.clickableObjects) {
              if (obj.contains(mouse_3)) {
                  gotValidClick = true;
                  mouse_3.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_3.clicked_name.push(null);
          }
          _mouseXYs = mouse_3.getPos();
          mouse_3.x.push(_mouseXYs[0]);
          mouse_3.y.push(_mouseXYs[1]);
          mouse_3.leftButton.push(_mouseButtons[0]);
          mouse_3.midButton.push(_mouseButtons[1]);
          mouse_3.rightButton.push(_mouseButtons[2]);
          mouse_3.time.push(mouse_3.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play_4* updates
    if (t >= 0.0 && play_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play_4.tStart = t;  // (not accounting for frame time here)
      play_4.frameNStart = frameN;  // exact frame index
      
      play_4.setAutoDraw(true);
    }
    
    
    // if play_4 is active this frame...
    if (play_4.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Start_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function Start_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Start_2' ---
    Start_2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Start_2.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_3.x', mouse_3.x);
    psychoJS.experiment.addData('mouse_3.y', mouse_3.y);
    psychoJS.experiment.addData('mouse_3.leftButton', mouse_3.leftButton);
    psychoJS.experiment.addData('mouse_3.midButton', mouse_3.midButton);
    psychoJS.experiment.addData('mouse_3.rightButton', mouse_3.rightButton);
    psychoJS.experiment.addData('mouse_3.time', mouse_3.time);
    psychoJS.experiment.addData('mouse_3.clicked_name', mouse_3.clicked_name);
    
    // the Routine "Start_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function Instrucciones_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Instrucciones_2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    Instrucciones_2Clock.reset();
    routineTimer.reset();
    Instrucciones_2MaxDurationReached = false;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse
    // current position of the mouse:
    mouse.x = [];
    mouse.y = [];
    mouse.leftButton = [];
    mouse.midButton = [];
    mouse.rightButton = [];
    mouse.time = [];
    mouse.clicked_name = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Instrucciones_2.started', globalClock.getTime());
    Instrucciones_2MaxDuration = null
    // keep track of which components have finished
    Instrucciones_2Components = [];
    Instrucciones_2Components.push(Indicaciones_3);
    Instrucciones_2Components.push(mouse);
    Instrucciones_2Components.push(play_3);
    Instrucciones_2Components.push(Arriba_Start_2);
    Instrucciones_2Components.push(Abajo_Start_2);
    
    Instrucciones_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function Instrucciones_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Instrucciones_2' ---
    // get current time
    t = Instrucciones_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Indicaciones_3* updates
    if (t >= 0.0 && Indicaciones_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Indicaciones_3.tStart = t;  // (not accounting for frame time here)
      Indicaciones_3.frameNStart = frameN;  // exact frame index
      
      Indicaciones_3.setAutoDraw(true);
    }
    
    
    // if Indicaciones_3 is active this frame...
    if (Indicaciones_3.status === PsychoJS.Status.STARTED) {
    }
    
    // *mouse* updates
    if (t >= 0.0 && mouse.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse.tStart = t;  // (not accounting for frame time here)
      mouse.frameNStart = frameN;  // exact frame index
      
      mouse.status = PsychoJS.Status.STARTED;
      mouse.mouseClock.reset();
      prevButtonState = mouse.getPressed();  // if button is down already this ISN'T a new click
    }
    
    // if mouse is active this frame...
    if (mouse.status === PsychoJS.Status.STARTED) {
      _mouseButtons = mouse.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse.clickableObjects = eval(play_2)
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse.clickableObjects)) {
              mouse.clickableObjects = [mouse.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse.clickableObjects) {
              if (obj.contains(mouse)) {
                  gotValidClick = true;
                  mouse.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse.clicked_name.push(null);
          }
          _mouseXYs = mouse.getPos();
          mouse.x.push(_mouseXYs[0]);
          mouse.y.push(_mouseXYs[1]);
          mouse.leftButton.push(_mouseButtons[0]);
          mouse.midButton.push(_mouseButtons[1]);
          mouse.rightButton.push(_mouseButtons[2]);
          mouse.time.push(mouse.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *play_3* updates
    if (t >= 0.0 && play_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      play_3.tStart = t;  // (not accounting for frame time here)
      play_3.frameNStart = frameN;  // exact frame index
      
      play_3.setAutoDraw(true);
    }
    
    
    // if play_3 is active this frame...
    if (play_3.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Arriba_Start_2* updates
    if (t >= 0.0 && Arriba_Start_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Arriba_Start_2.tStart = t;  // (not accounting for frame time here)
      Arriba_Start_2.frameNStart = frameN;  // exact frame index
      
      Arriba_Start_2.setAutoDraw(true);
    }
    
    
    // if Arriba_Start_2 is active this frame...
    if (Arriba_Start_2.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Abajo_Start_2* updates
    if (t >= 0.0 && Abajo_Start_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Abajo_Start_2.tStart = t;  // (not accounting for frame time here)
      Abajo_Start_2.frameNStart = frameN;  // exact frame index
      
      Abajo_Start_2.setAutoDraw(true);
    }
    
    
    // if Abajo_Start_2 is active this frame...
    if (Abajo_Start_2.status === PsychoJS.Status.STARTED) {
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Instrucciones_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function Instrucciones_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Instrucciones_2' ---
    Instrucciones_2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Instrucciones_2.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse.x', mouse.x);
    psychoJS.experiment.addData('mouse.y', mouse.y);
    psychoJS.experiment.addData('mouse.leftButton', mouse.leftButton);
    psychoJS.experiment.addData('mouse.midButton', mouse.midButton);
    psychoJS.experiment.addData('mouse.rightButton', mouse.rightButton);
    psychoJS.experiment.addData('mouse.time', mouse.time);
    psychoJS.experiment.addData('mouse.clicked_name', mouse.clicked_name);
    
    // the Routine "Instrucciones_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function Trial_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Trial_2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    Trial_2Clock.reset();
    routineTimer.reset();
    Trial_2MaxDurationReached = false;
    // update component parameters for each repeat
    Instruccion.setImage(Flechas_Random);
    Cuenta_tarea2.setText(contador);
    // Run 'Begin Routine' code from code_4
    contador = (contador + 1);
    respuesta = "";
    respuesta_recibida = false;
    esp32.reset_input_buffer();
    
    psychoJS.experiment.addData('Trial_2.started', globalClock.getTime());
    Trial_2MaxDuration = null
    // keep track of which components have finished
    Trial_2Components = [];
    Trial_2Components.push(Izquierda_2);
    Trial_2Components.push(Derecha);
    Trial_2Components.push(Arriba);
    Trial_2Components.push(Abajo);
    Trial_2Components.push(Instruccion);
    Trial_2Components.push(Cuenta_tarea2);
    
    Trial_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}

function Trial_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Trial_2' ---
    // get current time
    t = Trial_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *Izquierda_2* updates
    if (t >= 0.0 && Izquierda_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Izquierda_2.tStart = t;  // (not accounting for frame time here)
      Izquierda_2.frameNStart = frameN;  // exact frame index
      
      Izquierda_2.setAutoDraw(true);
    }
    
    
    // if Izquierda_2 is active this frame...
    if (Izquierda_2.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Derecha* updates
    if (t >= 0.0 && Derecha.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Derecha.tStart = t;  // (not accounting for frame time here)
      Derecha.frameNStart = frameN;  // exact frame index
      
      Derecha.setAutoDraw(true);
    }
    
    
    // if Derecha is active this frame...
    if (Derecha.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Arriba* updates
    if (t >= 0 && Arriba.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Arriba.tStart = t;  // (not accounting for frame time here)
      Arriba.frameNStart = frameN;  // exact frame index
      
      Arriba.setAutoDraw(true);
    }
    
    
    // if Arriba is active this frame...
    if (Arriba.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Abajo* updates
    if (t >= 0 && Abajo.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Abajo.tStart = t;  // (not accounting for frame time here)
      Abajo.frameNStart = frameN;  // exact frame index
      
      Abajo.setAutoDraw(true);
    }
    
    
    // if Abajo is active this frame...
    if (Abajo.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Instruccion* updates
    if (t >= 0.0 && Instruccion.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Instruccion.tStart = t;  // (not accounting for frame time here)
      Instruccion.frameNStart = frameN;  // exact frame index
      
      Instruccion.setAutoDraw(true);
    }
    
    
    // if Instruccion is active this frame...
    if (Instruccion.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *Cuenta_tarea2* updates
    if (t >= 0.0 && Cuenta_tarea2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      Cuenta_tarea2.tStart = t;  // (not accounting for frame time here)
      Cuenta_tarea2.frameNStart = frameN;  // exact frame index
      
      Cuenta_tarea2.setAutoDraw(true);
    }
    
    
    // if Cuenta_tarea2 is active this frame...
    if (Cuenta_tarea2.status === PsychoJS.Status.STARTED) {
    }
    
    // Run 'Each Frame' code from code_4
    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    if ((! respuesta_recibida)) {
        if ((esp32.in_waiting > 0)) {
            try {
                dato = esp32.readline().decode({"errors": "ignore"}).strip();
                console.log(dato);
                if (_pj.in_es6(dato, ["ARRIBA", "ABAJO", "IZQUIERDA", "DERECHA"])) {
                    respuesta = dato;
                    respuesta_recibida = true;
                    continueRoutine = false;
                }
            } catch(e) {
            }
        }
    }
    if ((respuesta_recibida && (! feedbackPlayed))) {
        if ((respuesta === Correct)) {
            feedback.setSound("correct.mp3");
        } else {
            feedback.setSound("incorrect.mp3");
        }
        feedback.play();
        feedbackPlayed = true;
    }
    if ((feedbackPlayed && (t >= (feedbackStart + 0.5)))) {
        continueRoutine = false;
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    Trial_2Components.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function Trial_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Trial_2' ---
    Trial_2Components.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('Trial_2.stopped', globalClock.getTime());
    // Run 'End Routine' code from code_4
    psychoJS.experiment.addData("respuesta", respuesta);
    
    // the Routine "Trial_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}

async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  // Run 'End Experiment' code from code_6
  esp32.close();
  
  // Run 'End Experiment' code from code_4
  esp32.close();
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
