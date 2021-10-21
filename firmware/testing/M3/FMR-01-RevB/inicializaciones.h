void iniTimer2();


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
