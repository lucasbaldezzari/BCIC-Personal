/*
* usart.c
*
* Created : 15-08-2020 09:34:44 PM
* Author  : Arnab Kumar Das
* Website : www.ArnabKumarDas.com
*/

#define F_CPU 16000000UL // Defining the CPU Frequency

#define FREC_INTERRUPT  5000 //en Hz
#define PRE_SCALER      64UL

#include <avr/io.h>      // Contains all the I/O Register Macros
#include <util/delay.h>  // Generates a Blocking Delay
#include <avr/interrupt.h> // Contains all interrupt vectors

#define USART_BAUDRATE 19200 // Desired Baud Rate
#define BAUD_PRESCALER (((F_CPU / (USART_BAUDRATE * 16UL))) - 1)

#define ASYNCHRONOUS (0<<UMSEL00) // USART Mode Selection

#define DISABLED    (0<<UPM00)
#define EVEN_PARITY (2<<UPM00)
#define ODD_PARITY  (3<<UPM00)
#define PARITY_MODE  DISABLED // USART Parity Bit Selection

#define ONE_BIT (0<<USBS0)
#define TWO_BIT (1<<USBS0)
#define STOP_BIT ONE_BIT      // USART Stop Bit Selection

#define FIVE_BIT  (0<<UCSZ00)
#define SIX_BIT   (1<<UCSZ00)
#define SEVEN_BIT (2<<UCSZ00)
#define EIGHT_BIT (3<<UCSZ00)
#define DATA_BIT   EIGHT_BIT  // USART Data Bit Selection

#define RX_COMPLETE_INTERRUPT         (1<<RXCIE0)
#define DATA_REGISTER_EMPTY_INTERRUPT (1<<UDRIE0)

void iniTimer0(int frecTimer);
void iniTimer2();
void iniUART();

void iniTimer0(int frecTimer)
{
//Seteamos el Timer0 para que trabaje a 5000Hz = 0.2ms
  TCCR0A = 0;// pongo a cero el registro de control del timer1
  TCCR0B = 0;// Lo mismo para el TCCR0B
  TCNT0  = 0;//initialize counter value to 0
  
  // turn on CTC mode
  TCCR0A |= (1 << WGM01);//Ponemos un 1 en el Bit WGM01 del registro TCCR0A - Modo CTC (ver página 107)
  // Seteamos el PreScaler en 64 (ver página 109 de la hoja de datos)
  TCCR0B |= (0 << CS02) | (1 << CS01) | (1 << CS00);
  int preScaler = 64UL;
  // Cargamos el comparador del Timer0 para que nos de una interrupción aproximadamente de 0.1ms
  unsigned char comparador = ((F_CPU/(PRE_SCALER*frecTimer)) - 1);
  OCR0A = comparador;
  //OCR0A = 49;// = (16MHz/(preScaler*frecuencia de Interrupción))-1

  //Habilito la interrupción (ver pagina 110 de hoja de datos)
  TIMSK0 |= (1 << OCIE0A);
  }

void iniTimer2()
{
  TCCR2A = 0;// pongo a cero el registro de control del timer2 (pagina 155)
  TCCR2B = 0;// idem para TCCR2B
  TCNT2  = 0;//valor inicial del contador en 0
  //seteamos el timer2 para interrupción cada 1ms, que es el máximo tiempo que podemos alcanzar
  OCR2A = 249; // = (16MHz/(preScaler*frecuencia de Interrupción))-1
  //Modo Contador (Clear Timer on Compare Match (CTC))
  TCCR2A |= (1 << WGM21);
  // Seteo el preescaler en 64
  TCCR2B |= (1 << CS22) | (0 << CS21) | (0 << CS20);   
  // Habilito la comparación
  TIMSK2 |= (1 << OCIE2A);	//ver pagina 160
}

void iniUART()
{
  UBRR0 = BAUD_PRESCALER; // baud rate of 9600bps
  UCSR0C |= (0 << UMSEL01) | (0 << UMSEL00) | (1 << UPM01) | (0 << UPM00) | (0 << USBS0) | (1 << UCSZ02) | (1 << UCSZ01) | (1 << UCSZ00); //8 bits de dato (ver página 197 de la hoja de datos)
  UCSR0B |= (1 << RXEN0) | (1 << TXEN0) | (1 << RXCIE0); //se habilitan interrupciones para la transmisión y la recepción
}
