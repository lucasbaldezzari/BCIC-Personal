/*
Ejemplo de configuración de Interrupciónes usando los Timer0, Timer1 y Timer2 de nuestra placa Arduino 1.
*/
#include "definiciones.h"
#include "inicializaciones.h"
#include "funciones.h"


/*Declaración de variables generales*/

char uartIn = 0; //para recepción de UART
char bufferSize = 6;
unsigned int incDataFromPC[6]; //variable para almacenar datos provenientes de la PC
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

unsigned char outputDataToPC[4]; //variable para enviar datos a la PC
char buffOutDataPCSize = 4;
char buffOutDataPCIndex = 0;

unsigned char outputDataToRobot[4]; //variable para enviar datos al robot
char buffOutDataRobotSize = 4;
char buffOutDataRobotIndex = 0;

unsigned char incDataFromRobot[4]; //variable para recibir datos del robot
char incDataFromRobotSize = 4;
char incDataFromRobotIndex = 0;

char sessionState = 0; //Sesión sin iniciar

String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete
char inDataLen = 6;

/* Declaración de variables para control de estímulos*/
int frecTimer = 5000; //en Hz

char estimIzq = 11;
bool estimIzqON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimIzq = 11;
int acumEstimIzq = 0;
const int estimIzqMaxValue = (1/float(frecEstimIzq))*frecTimer;

//estímulo derecho
char estimDer = 7;
bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimDer = 11;
int acumEstimDer = 0;
const int estimDerMaxValue = (1/float(frecEstimDer))*frecTimer;

char flagLED = 13; //led de testeo

//unsigned int timeON = 4*frecTimer; //Los estímulos estarán encendidos por 4 segundos 
//int timeOFF = 6*frecTimer; //Los estímulos estarán apagados por 2 segundos
//const unsigned int maxTrialTime = (timeON + timeOFF); 
char stimuli = ON; //variable de control de estímulos globales
int acumuladorStimuliON = 0;
int trialNumber = 1;

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(estimIzq,OUTPUT);
  pinMode(estimDer,OUTPUT);
  pinMode(flagLED,OUTPUT);
  iniTimer0(); //inicio timer 0
  Serial.begin(19200); //iniciamos comunicación serie
  interrupts();//Habilito las interrupciones

  //Cargo datos en buffer de outputDataToPC para simular que el robot ve algunos obstáculos
  outputDataToPC[FORWARD_INDEX] = OBSTACULO_DETECTADO;
  outputDataToPC[LEFT_INDEX] = OBSTACULO_DETECTADO;
  outputDataToPC[RIGHT_INDEX] = SIN_OBSTACULO;
  outputDataToPC[BACK_INDEX] = SIN_OBSTACULO;  
}

void loop(){}

void serialEvent()
{
if (Serial.available() > 0) 
  {
    int val = char(Serial.read()) - '0';

    if(val != 2)
    {
      incDataFromPC[bufferIndex] = val;
      checkMessage();        
      Serial.write("OK\n");
    }

    if (val == 2) //Solicitud para enviar estado del robot
    {
      for(int index = 0; index < buffOutDataPCSize; index++)
        {
          Serial.write(outputDataToPC[index]);
        }
        Serial.write("\n");
      }
    }
};

ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0. Se configuró para 0.1ms
{
  if(sessionState) stimuliControl(); //Si la sesión comenzó, empezamos a generar los estímulos  
  else
  {
    digitalWrite(estimIzq,0);
    digitalWrite(estimDer,0);
    digitalWrite(flagLED,1);
  }
};

void stimuliControl()
{
  
  acumuladorStimuliON++;
  switch(stimuli)
  {
    case ON:
    //control estímulo izquierdo
      if (++acumEstimIzq >= estimIzqMaxValue)
      {
        estimIzqON = !estimIzqON;
        digitalWrite(estimIzq,estimIzqON);
        acumEstimIzq = 0; 
      } 

    //control estímulo derecho
      if (++acumEstimDer >= estimDerMaxValue)
      {
        estimDerON = !estimDerON;
        digitalWrite(estimDer,estimDerON);
        acumEstimDer = 0; 
      } 
      break;

    case OFF:
      {
        //Apagamos estímulos y reiniciamos contadores
        digitalWrite(estimIzq,0);
        acumEstimIzq = 0;
        digitalWrite(estimDer,0);
        acumEstimDer = 0;
        trialNumber++; //Sumamos un nuevo trial
        acumuladorStimuliON = 0; //reiniciamos acumulador para temporizar cada trial
        acumEstimIzq = 0;
        acumEstimDer = 0;
      } 
      break;
  }
}

void checkMessage()
{
  switch(bufferIndex)
  {
    case 0:
      if (incDataFromPC[bufferIndex] == 1) sessionState = RUNNING;
      else sessionState = STOP;
      digitalWrite(flagLED,0);
      //else sessionState = STOP;
      break;
    case 1:
      if (incDataFromPC[bufferIndex] == 1) stimuli = ON;
      else stimuli = OFF;
      //else sessionState = STOP;
      break;
  }
  bufferIndex++;
  if (bufferIndex >= inDataLen) bufferIndex = 0;
};