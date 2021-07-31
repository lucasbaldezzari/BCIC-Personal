/*
Ejemplo de configuración de Interrupciónes usando los Timer0, Timer1 y Timer2 de nuestra placa Arduino 1.
*/
#include "definiciones.h"
#include "inicializaciones.h"
#include "funciones.h"

/*Declaración de variables generales*/

char uartIn = 0; //para recepción de UART
char bufferSize = 5;
unsigned int incomingData[5];
char bufferIndex = 0;
bool sendDataFlag = 0;
bool newMessage = false;

char mensajeRecibido = 1;

char sessionState = 0; //Sesión sin iniciar

String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete
char inDataLen = 5;

/* Declaración de variables para control de estímulos*/


int frecTimer = 10000; //en Hz

char estimIzq = 13;
bool estimIzqON = 0;//Esado que define si el LED se apgará o prenderá.
int frecEstimIzq = 14;
int acumEstimIzq = 0;

//estímulo derecho
char estimDer = 11;
bool estimDerON = 0;//Esado que define si el LED se apgará o prenderá.

int frecEstimDer = 16;
int acumEstimDer = 0;
const int estimIzqMaxValue = 714;//(1/float(frecEstimIzq))*frecTimer;

unsigned int timeON = 4*frecTimer; //Los estímulos estarán encendidos por 4 segundos 
int timeOFF = 2*frecTimer; //Los estímulos estarán apagados por 2 segundos
const unsigned int maxTrialTime = (timeON + timeOFF); 
char stimuli = ON; //variable de control de estímulos globales
int acumuladorStimuliON = 0;
int trialNumber = 1;

//FUNCION SETUP
void setup()
{
  noInterrupts();//Deshabilito todas las interrupciones
  pinMode(estimIzq,OUTPUT);
  pinMode(estimDer,OUTPUT);
  // reservo 5 bytes para el string de datos
  //iniUART();
  iniTimer0();
  Serial.begin(19200);
  interrupts();//Habilito las interrupciones
}

void loop(){}

void serialEvent()
{
if (Serial.available() > 0) 
  {
    int val = char(Serial.read()) - '0';
    incomingData[bufferIndex] = val;
    checkMessage();    
    Serial.write("OK\n");
    //digitalWrite(estimDer,1);
    }
};

ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0. Se configuró para 0.1ms
{
  if(sessionState) stimuliControl(); //Si la sesión comenzó, empezamos a generar los estímulos  
  else
  {
    digitalWrite(estimIzq,0);
    digitalWrite(estimDer,0);
  }
  //if(!sessionState) digitalWrite(estimDer,1);
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

      //Control tiempo de estímulos encendidos
      if (acumuladorStimuliON >= timeON)
      {
        //Fin de tiempo estímulos encendidos
        stimuli = OFF;
        //Apagamos todos los leds y reiniciamos sus contadores
        digitalWrite(estimIzq,0);
        acumEstimIzq = 0;
      } 
      break;

    case OFF:
      //Control tiempo de estímulos apagados para terminar el trial
      if (acumuladorStimuliON >= maxTrialTime)
      {
        //comenzamos un nuevo trial y debemos encender los estímulos
        stimuli = ON;
        trialNumber++; //Sumamos un nuevo trial
        acumuladorStimuliON = 0; //reiniciamos acumulador para temporizar cada trial
        sendDataFlag = 1;
      } 
      break;
  }
}

void checkMessage()
{
  switch(bufferIndex)
  {
    case 0:
      if (incomingData[bufferIndex] == 1) sessionState = RUNNING;
      else sessionState = STOP;
      //else sessionState = STOP;
      
      break;
  }
  bufferIndex++;
  if (bufferIndex >= inDataLen) bufferIndex = 0;
};
