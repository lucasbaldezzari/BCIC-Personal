#include <SoftwareSerial.h>
#include "inicializaciones.h"

#define   FRENADO    1
#define   MOVIENDO   0

SoftwareSerial BT(A0, A1);  // pin 10 TX, pin 11 RX

char Dt = 0;

char ledRojo = 11;
char ledVerde = 12;
char ledAzul = 13;

int temporizador = 500; //para 100ms cFRENADOsiderando la frecuencia de interrupción del timer2
int acum = 0;
bool estado = 0;

char flagMoviendo = FRENADO;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0;

// Motor A
int ENA = 5;
int IN1 = 6;
int IN2 = 7;

// Motor B
int ENB = 10;
int IN3 = 8;
int IN4 = 9;


void setup() {
  noInterrupts();//Deshabilito todas las interrupciFRENADOes
  //Motores
  pinMode (ENA, OUTPUT);
  pinMode (ENB, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (ledRojo, OUTPUT);
  pinMode (ledVerde, OUTPUT);
  pinMode (ledAzul, OUTPUT);

  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
  BT.begin(9600);

  iniTimer2();
  delay(1000);
  digitalWrite(ledRojo, 0);
  interrupts();//Habilito las interrupciFRENADOes
}

void loop() {}

/*Rutina interrupciòn Timer0*/
ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2
{
  while (BT.available())
  {
    Dt = BT.read();
    giveAnOrder();//damos una orden al vehículo
    sendBTMessage();
  }

  switch (flagMoviendo)
  {
    case MOVIENDO:
      if (++acum > temporizador)
      {
        estado = !estado;
        digitalWrite(ledRojo, estado);
        acum = 0;
      }
      break;

    case FRENADO:
      if (++acum > temporizador)
      {
        estado = !estado;
        digitalWrite(ledAzul, estado);
        acum = 0;
      }
      break;
  }

};

void Adelante()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 100);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
}

void Retroceso()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 100);
  digitalWrite (IN1, HIGH );
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW );
  digitalWrite (IN4, HIGH);
}

void Derecha()
{
  analogWrite (ENA, 0);
  analogWrite (ENB, 100);
  digitalWrite (IN1, LOW );
  digitalWrite (IN2, LOW );
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
}

void Izquierda()
{
  analogWrite (ENA, 100);
  analogWrite (ENB, 0);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, LOW );
  digitalWrite (IN4, LOW );
}

void Stop()
{
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, LOW);
  analogWrite (ENA, 0); //motores parados
  analogWrite (ENB, 0); //motores parados
}

void giveAnOrder()
{
  if ( ( (Dt >> 1) & 0b00000001) == 0 ) //Si los estímulos se apagarFRENADO, podemos mover.
  {
    flagMoviendo = MOVIENDO;
    digitalWrite(ledAzul, 0);
    acum = 0;
    if ( ((Dt >> 2) & mascaraComando) == 1) Adelante();
    else if ( ((Dt >> 2) & mascaraComando) == 2) Izquierda();
    else if ( ((Dt >> 2) & mascaraComando) == 3) Retroceso();
    else if ( ((Dt >> 2) & mascaraComando) == 4 ) Derecha();
    else    {
      Stop();
    }
  }

  else
  {
    Stop();
    flagMoviendo = FRENADO;
    digitalWrite(ledRojo, 0);
    acum = 0;
  }

  if ( ( (Dt >> 0) & 0b00000001) == 0 ) //Sesión frenada
  {
    Stop();
    flagMoviendo = FRENADO;
    digitalWrite(ledRojo, 0);
    acum = 0;
  }

}

void sendBTMessage()
{
  BT.write(0b00000010);
}
