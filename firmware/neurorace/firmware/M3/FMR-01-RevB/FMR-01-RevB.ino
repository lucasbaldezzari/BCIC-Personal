#include <SoftwareSerial.h>  // 
#include "inicializaciones.h"

SoftwareSerial BT(12, 13);  // pin 10 TX, pin 11 RX

char Dt = 0; 

volatile unsigned int cuenta = 0;
bool estado = false;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0;

 // Motor A
int ENA = 10;
int IN1 = 9;
int IN2 = 8;

// Motor B
int ENB = 5;
int IN3 = 7;
int IN4 = 6;  
   

void setup(){

  //Motores
  pinMode (ENA, OUTPUT);
  pinMode (ENB, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
 
  analogWrite (ENA, 255); //Velocidad motor A
  analogWrite (ENB, 255); //Velocidad motor A
  digitalWrite(13,0);
  BT.begin(9600);    
  
  iniTimer0();
}

void loop(){}

/*Rutina interrupciòn Timer0*/
ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0.
{
  //estado = !estado;
  //digitalWrite(13,estado); //para chequear frecuencia de interrupción

  while (BT.available())
  {     
    Dt = BT.read();   
    void giveAnOrder();//damos una orden al vehículo
    sendBTMessage();
  }
};

void Adelante()
{
  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
}
void Retroceso()
{
  digitalWrite (IN1,LOW );
  digitalWrite (IN2,HIGH );
  digitalWrite (IN3,LOW );
  digitalWrite (IN4,HIGH );
}
void Derecha()
{ 
  digitalWrite (IN1,LOW );
  digitalWrite (IN2,HIGH );
  digitalWrite (IN3,HIGH);
  digitalWrite (IN4,LOW);
}
void Izquierda()
{ 
  digitalWrite (IN1,HIGH);
  digitalWrite (IN2,LOW);
  digitalWrite (IN3,LOW );
  digitalWrite (IN4,HIGH );
}
void Stop()
{
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, LOW);
}

void giveAnOrder()
{
  if ( ( (Dt>>1)&0b00000001) == 0 ) //Si los estímulos se apagaron, podemos mover.
  {
         if ( ((Dt >> 2) & mascaraComando) == 1) Adelante();
    else if ( ((Dt >> 2) & mascaraComando) == 2){Izquierda();}
    else if ( ((Dt >> 2) & mascaraComando) == 3){Retroceso();}
    else if ( ((Dt >> 2) & mascaraComando) == 4 ){Derecha();}
    else    {Stop();}
  }
  else Stop();
}

void sendBTMessage()
{
  BT.write(0b00000010);
  }
