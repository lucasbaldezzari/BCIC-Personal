/******************************************************************
            VERSIÓN FMR-001 Rev A
******************************************************************/

#include "definiciones.h"
#include "inicializaciones.h"
#include "funciones.h"
#include<SoftwareSerial.h>

/******************************************************************
  Declaración de variables para comunicación
/******************************************************************/

char uno = 1;
byte backMensaje = 0b00000000; 

char inBuffDataFromPC = 3;
unsigned char incDataFromPC[3]; //variable para almacenar datos provenientes de la PC
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

unsigned char internalStatus[4]; //variable para enviar datos a la PC
char internalStatusBuff = 4;

unsigned char outputDataToRobot[4]; //variable para enviar datos al robot
char buffOutDataRobotSize = 4;
char buffOutDataRobotIndex = 0;

unsigned char incDataFromRobot[4]; //variable para recibir datos del robot
char incDataFromRobotSize = 4;
char incDataFromRobotIndex = 0;

/******************************************************************
  Variables para el control de flujo de programa
******************************************************************/
char sessionState = 0; //Sesión sin iniciar
char LEDVerde = 8; 
//char LEDTesteo = 5; //led de testeo

/******************************************************************
  Declaración de variables para control de estímulos
******************************************************************/

int frecTimer = 5000; //en Hz. Frecuencia de interrupción del timer.

//estímulo adelante
char estimAd = 10;
bool estimAdON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimAd = 14;
int acumEstimAd = 0;
const int estimAdMaxValue = (1/float(frecEstimAd))*frecTimer;

//estímulo atras
char estimAt = 13;
bool estimAtON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimAt = 26;
int acumEstimAt = 0;
const int estimAtMaxValue = (1/float(frecEstimAt))*frecTimer;

//estímulo derecho
char estimDer = 11;
bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimDer = 20;
int acumEstimDer = 0;
const int estimDerMaxValue = (1/float(frecEstimDer))*frecTimer;

//estímulo izquierdo
char estimIz = 12;
bool estimIzON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimIz = 17;
int acumEstimIz = 0;
const int estimIzMaxValue = (1/float(frecEstimIz))*frecTimer;
//
////estímulo rotacion derecha
//char estimDer = 12;
//bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
//int frecEstimDer = 3;
//int acumEstimDer = 0;
//const int estimDerMaxValue = (1/float(frecEstimDer))*frecTimer;
//
////estímulo rotacion izquierda
//char estimDer = 12;
//bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
//int frecEstimDer = 3;
//int acumEstimDer = 0;
//const int estimDerMaxValue = (1/float(frecEstimDer))*frecTimer;



/*Implementar lo siguiente
//estímulo adelante
//estímulo atrás
*/

char stimuli = ON; //variable golbal para control de estímulos
int acumuladorStimuliON = 0;
int trialNumber = 1;

/******************************************************************
  Variables control de movimiento
******************************************************************/

char movimiento = 0; //Robot en STOP
int estado = 0;

SoftwareSerial BTMaestro(2,3); //TX, RX

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(estimIz,OUTPUT);
  pinMode(estimDer,OUTPUT);
  pinMode(estimAd,OUTPUT);
  pinMode(estimAt,OUTPUT);
  
  pinMode(LEDVerde,OUTPUT);
  digitalWrite(LEDVerde,0);
  //pinMode(LEDTesteo,OUTPUT);
  //pinMode(3,OUTPUT);
  iniTimer2(); //inicio timer 2
  Serial.begin(19200); //iniciamos comunicación serie
  BTMaestro.begin(9600);//comunicacion al BT
  interrupts();//Habilito las interrupciones
}

void loop(){}

void serialEvent()
{
if (Serial.available() > 0) 
  {
    char val = char(Serial.read()) - '0';
    checkMessage(val); //chequeamos mensaje entrante        
    Serial.println(backMensaje);
//    for(int index = 0; index < internalStatusBuff; index++) //enviamos estado 
//      {Serial.write(internalStatus[index]);}
//      Serial.write("\n");
  }
};

ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer0.
{
  if(sessionState == SESSION_RUNNING) stimuliControl(); //Si la sesión comenzó, empezamos a generar los estímulos  
  else
  {
    //apago estímulos
    digitalWrite(estimIz,0);
    digitalWrite(estimDer,0);
    digitalWrite(estimAd,0);
    digitalWrite(estimAt,0);
    
    //digitalWrite(LEDTesteo,1);
  }

    if(BTMaestro.available()) //Si tenemos un mensaje por bluetooth lo leemos
  {
    backMensaje = BTMaestro.read();
    //checkBTMessage(backMensaje);
    //estado = !estado;
    //digitalWrite(LEDVerde,estado);
  }
};

void stimuliControl()
{
  acumuladorStimuliON++;
  switch(stimuli)
  {
    case ON:
    //control estímulo izquierdo
      if (++acumEstimIz >= estimIzMaxValue)
      {
        if ((backMensaje & 0b00000010) != 0b00000010){ 
        estimIzON = !estimIzON;
        digitalWrite(estimIz,estimIzON);
        } 
        acumEstimIz = 0; 
      } 
      
    //control estímulo derecho
      if (++acumEstimDer >= estimDerMaxValue)
      {
        if ((backMensaje & 0b00000100) != 0b00000100){
        estimDerON = !estimDerON;
        digitalWrite(estimDer,estimDerON);
        }
        acumEstimDer = 0; 
      } 
      

    //control estímulo adelante
      if (++acumEstimAd >= estimAdMaxValue)
        {
        if ((backMensaje & 0b00000001) != 0b00000001){ 
        estimAdON = !estimAdON;
        digitalWrite(estimAd,estimAdON);
        }
        acumEstimAd = 0; 
      } 
      
      
    //control estímulo atras
      if (++acumEstimAt >= estimAtMaxValue)
      {
        if ((backMensaje & 0b00001000) != 0b00001000){
        estimAtON = !estimAtON;
        digitalWrite(estimAt,estimAtON);
        }
        acumEstimAt = 0; 
      } 
      break;

    case OFF:
      {
        //Apagamos estímulos y reiniciamos contadores
        digitalWrite(estimIz,0);
        acumEstimIz = 0;
        digitalWrite(estimDer,0);
        acumEstimDer = 0;
        digitalWrite(estimAd,0);
        acumEstimAd = 0;
        digitalWrite(estimAt,0);
        acumEstimAt = 0;
        trialNumber++; //Sumamos un nuevo trial
        acumuladorStimuliON = 0; //reiniciamos acumulador para temporizar cada trial
        acumEstimIz = 0;
        acumEstimDer = 0;
        acumEstimAd = 0;
        acumEstimAt = 0;
      } 
      break;
  }
}

void checkMessage(byte val)
{
  incDataFromPC[bufferIndex] = val;
  switch(bufferIndex)
  {
    case 0:
      if (incDataFromPC[bufferIndex] == 1) sessionState = SESSION_RUNNING;
      else sessionState = SESSION_STOP;
      //digitalWrite(LEDTesteo,0);
      
      //else sessionState = STOP;
      break;
    case 1:
      if (incDataFromPC[bufferIndex] == 1) stimuli = ON;
      else stimuli = OFF;
      
      break;

    case 2: //indica hacia donde se debe mover el vehículo
      byte comando = incDataFromPC[bufferIndex];
      //if(comando == 3) digitalWrite(LEDVerde,1);
       //Es mejor hacer sendCommand(incDataFromPC[bufferIndex])
      break;          
  }
  bufferIndex++;
  if (bufferIndex >= inBuffDataFromPC) //hemos recibido todos los bytes desde la PC
  {
    sendCommand();
    bufferIndex = 0;
  }
};

/*
Función: sendCommand()
- Se usa para enviar un comando al vehículo robótico y recibir el estado del mismo.
*/
void sendCommand()
{
    estado = !estado;
    digitalWrite(LEDVerde,estado);
    byte mensaje = (incDataFromPC[0])|(incDataFromPC[1]<<1)|(incDataFromPC[2]<<2);//Armamos el byte
    BTMaestro.write(mensaje); //enviamos byte por bluetooth
    //BTMaestro.write(uno);
}
