void iniTimer0();

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
  // Seteo el preescaler en 64
  TCCR2B |= (1 << CS22) | (0 << CS21) | (0 << CS20);   
  //seteamos el timer2 para interrupción cada 1ms, que es el máximo tiempo que podemos alcanzar
  OCR2A = 49; // = (16MHz/(preScaler*frecuencia de Interrupción))-1
  //Modo Contador (Clear Timer on Compare Match (CTC))
  TCCR2A |= (1 << WGM21);

  // Habilito la comparación
  TIMSK2 |= (1 << OCIE2A);	//ver pagina 160
}
