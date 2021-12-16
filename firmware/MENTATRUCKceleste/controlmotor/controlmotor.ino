#include <SoftwareSerial.h>
#include "inicializaciones.h"

#define   FRENADO    1
#define   MOVIENDO   0


SoftwareSerial BTone(A3, A4);  // TX, RX

char Dt = 0;
int PWM = 100;
int PWM2 = 150;

char ledRojo = A0;
char ledVerde = A5;
char ledAzul = A1;

int temporizador = 500; //para 100ms cFRENADOsiderando la frecuencia de interrupción del timer2
int acum = 0;
bool estado = 0;
byte obstaculos = 0b00000000;

char flagMoviendo = FRENADO;

byte mascaraComando = 0b00000111;

unsigned int acumulador = 0;
int EN = 10;

// Motor A
int IN1 = 6;
int IN2 = 7;

// Motor B
int IN3 = 8;
int IN4 = 9;

// Motor C
int IN5 = 2;
int IN6 = 4;

// Motor D
int IN7 = 12;
int IN8 = 13;


void setup() {
  noInterrupts();//Deshabilito todas las interrupciFRENADOes
  //Motores
  pinMode (EN, OUTPUT);
  pinMode (IN1, OUTPUT);
  pinMode (IN2, OUTPUT);
  pinMode (IN3, OUTPUT);
  pinMode (IN4, OUTPUT);
  pinMode (IN5, OUTPUT);
  pinMode (IN6, OUTPUT);
  pinMode (IN7, OUTPUT);
  pinMode (IN8, OUTPUT);
  pinMode (ledRojo, OUTPUT);
  pinMode (ledVerde, OUTPUT);
  pinMode (ledAzul, OUTPUT);
  analogWrite (EN, 0); //motores parados
  BTone.begin(9600);
  Serial.begin(9600);


  iniTimer2();
  delay(1000);
  digitalWrite(ledRojo, 0);
  interrupts();//Habilito las interrupciFRENADOes
}

void loop() {
  //if (BTone.available()) digitalWrite(2,1);
  }

//void serialEvent()
//{
//  if (Serial.available())
//  {
//    obstaculos = Serial.read();  
//    if ((((obstaculos & 0b00000011) == 0b00000001) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001100) == 0b00000100) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00110000) == 0b00010000) && (((Dt >> 2) & mascaraComando) == 3)) || (((obstaculos & 0b11000000) == 0b01000000) && (((Dt >> 2) & mascaraComando) == 4))){
//      Stop();
//    }
//    if ((((obstaculos & 0b00000011) == 0b00000010) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001100) == 0b00001000) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00110000) == 0b00100000) && (((Dt >> 2) & mascaraComando) == 3)) || (((obstaculos & 0b11000000) == 0b10000000) && (((Dt >> 2) & mascaraComando) == 4))){
//      PWM = 125;
//    }
//    if ((((obstaculos & 0b00000011) == 0b00000011) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001100) == 0b00001100) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00110000) == 0b00110000) && (((Dt >> 2) & mascaraComando) == 3)) || (((obstaculos & 0b11000000) == 0b11000000) && (((Dt >> 2) & mascaraComando) == 4))){
//      PWM = 200;
//    }
//    BTone.write(obstaculos);
//  }
//}

/*Rutina interrupciòn Timer2*/
ISR(TIMER2_COMPA_vect)//Rutina interrupción Timer2
{
  if (BTone.available())
  {
    estado = !estado;
    digitalWrite(ledVerde,estado);
    Dt = BTone.read();    
    giveAnOrder();//damos una orden al vehículo
    //sendBTMessage();
    Serial.write(BTone.read());   
  }

  if (Serial.available())
  {
    obstaculos = Serial.read();  
    if ((((obstaculos & 0b00000001) == 0b00000001) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001000) == 0b00001000) && (((Dt >> 2) & mascaraComando) == 4)) || (((obstaculos & 0b00000010) == 0b00000010) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00000100) == 0b00000100) && (((Dt >> 2) & mascaraComando) == 3))){
      Stop();
    }
//    if ((((obstaculos & 0b00000011) == 0b00000010) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001100) == 0b00001000) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00110000) == 0b00100000) && (((Dt >> 2) & mascaraComando) == 3)) || (((obstaculos & 0b11000000) == 0b10000000) && (((Dt >> 2) & mascaraComando) == 4))){
//      PWM = 125;
//    }
//    if ((((obstaculos & 0b00000011) == 0b00000011) && (((Dt >> 2) & mascaraComando) == 1)) || (((obstaculos & 0b00001100) == 0b00001100) && (((Dt >> 2) & mascaraComando) == 2)) || (((obstaculos & 0b00110000) == 0b00110000) && (((Dt >> 2) & mascaraComando) == 3)) || (((obstaculos & 0b11000000) == 0b11000000) && (((Dt >> 2) & mascaraComando) == 4))){
//      PWM = 200;
//    }
//    else PWM = 255;
    BTone.write(obstaculos);
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
  analogWrite (EN, PWM);
  digitalWrite (IN1, LOW);//adelante iz
  digitalWrite (IN2, HIGH); //HIGH
  digitalWrite (IN3, LOW);// atras iz
  digitalWrite (IN4, HIGH); //HIGH
  digitalWrite (IN5, HIGH);//adelante der HIGH
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, LOW); // atras der
  digitalWrite (IN8, HIGH); //HIGH
}

void Retroceso()
{
  analogWrite (EN, PWM);
  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, HIGH);
  digitalWrite (IN7, HIGH);
  digitalWrite (IN8, LOW);
}

void Derecha()
{
  analogWrite (EN, PWM2);
  digitalWrite (IN1, HIGH);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, HIGH);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, HIGH);
  digitalWrite (IN7, LOW);  
  digitalWrite (IN8, HIGH); //HIGH
}

void Izquierda()
{
  analogWrite (EN, PWM2);
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, HIGH);
  digitalWrite (IN3, HIGH);
  digitalWrite (IN4, LOW);
  digitalWrite (IN5, HIGH);//
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, HIGH); //
  digitalWrite (IN8, LOW); //
}

void Stop()
{
  analogWrite (EN, 0); //motores parados
  digitalWrite (IN1, LOW);
  digitalWrite (IN2, LOW);
  digitalWrite (IN3, LOW);
  digitalWrite (IN4, LOW);
  digitalWrite (IN5, LOW);
  digitalWrite (IN6, LOW);
  digitalWrite (IN7, LOW);
  digitalWrite (IN8, LOW);
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
    else if ( ((Dt >> 2) & mascaraComando) == 4) Retroceso();
    else if ( ((Dt >> 2) & mascaraComando) == 3 ) Derecha();
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

//void sendBTMessage()
//{
//  BTone.write(0b00000010);
//}
