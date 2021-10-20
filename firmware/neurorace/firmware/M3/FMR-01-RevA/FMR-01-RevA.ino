#include <SoftwareSerial.h>  // 

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
  BT.begin(9600);    
  
 iniTimer0();
  
  //Motores
 pinMode (ENA, OUTPUT);
 pinMode (ENB, OUTPUT);
 pinMode (IN1, OUTPUT);
 pinMode (IN2, OUTPUT);
 pinMode (IN3, OUTPUT);
 pinMode (IN4, OUTPUT);
 
     analogWrite (ENA, 255); //Velocidad motor A
  analogWrite (ENB, 255); //Velocidad motor A
}

void loop(){
if (BT.available()){     
  Dt = BT.read();   

  if ( ( (Dt>>1)&0b00000001) == 0 )
  {
    if ( ((Dt >> 2) & mascaraComando) == 1) Adelante(); // 65 = A
      
      else if ( ((Dt >> 2) & mascaraComando) == 2){  // 66 = B
        Izquierda();
      }
      else if ( ((Dt >> 2) & mascaraComando) == 3){ // 67 = C
        Retroceso();
      }
      else if ( ((Dt >> 2) & mascaraComando) == 4 ){ // 68 = D
        Derecha();
      }
      else{
        Stop();
      }
    }
    else Stop();
  }
 }

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

void iniTimer0()
{
//Seteamos el Timer0 para que trabaje a 5000Hz = 0.2ms
  TCCR0A = 0;// pongo a cero el registro de control del timer1
  TCCR0B = 0;// Lo mismo para el TCCR0B
  TCNT0  = 0;//initialize counter value to 0
  
  // turn on CTC mode
  TCCR0A |= (1 << WGM01);//Ponemos un 1 en el Bit WGM01 del registro TCCR0A - Modo CTC (ver página 107)
  // Seteamos el PreScaler en 64 (ver página 109 de la hoja de datos)
  TCCR0B |= (0 << CS02) | (0 << CS01) | (1 << CS00);
  //int preScaler = 64UL;
  // Cargamos el comparador del Timer0 para que nos de una interrupción aproximadamente de 0.1ms
  //unsigned char comparador = ((F_CPU/(PRE_SCALER*frecTimer)) - 1);
  OCR0A = 159;
  //OCR0A = 49;// = (16MHz/(preScaler*frecuencia de Interrupción))-1

  //Habilito la interrupción (ver pagina 110 de hoja de datos)
  TIMSK0 |= (1 << OCIE0A);
  }

 ISR(TIMER0_COMPA_vect)//Rutina interrupción Timer0.
{
  estado = !estado;
  digitalWrite(13,estado);
};