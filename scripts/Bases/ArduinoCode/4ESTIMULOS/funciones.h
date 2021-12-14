void sendData(unsigned char dato);
void sendString(unsigned char &string, char bufferSize);
void checkMessage(unsigned char *data, char DataLen);

void sendData( unsigned char data)
  {
    /* Wait for empty transmit buffer */
    UCSR0B |= (1<<TXEN0)|(1<<UDRIE0);

    while ((UCSR0A & (1 << UDRE0)) == 0) {};
    /* Put data into buffer, sends the data */
    UDR0 = data;
  }

void sendString(unsigned char *stringData, char bufferSize)
  {
    char bufferIndex = 0;

    while (bufferIndex < bufferSize)
    {
      /* Wait for empty transmit buffer */
      sendData(*stringData);
      stringData++;
      //sendData(stringData[bufferIndex]);
      bufferIndex++;
    }
  }

void checkMessage(unsigned char *data, char DataLen)
{
  char index = 0;
  while (index < DataLen)
  {
    
  }
}
