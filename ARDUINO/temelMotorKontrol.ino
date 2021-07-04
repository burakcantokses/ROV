#include <Servo.h>
Servo ESC;

int joystickValue;

/*
 * Değiştirmen gereken yerler
 */
int PIN = 4;
int minESC = 1100;
int maxESC = 1900;

void setup() {
  /*
   * 1. parametre: PIN (4 olarak tanımladım)
   * 2. parametre: minimum değeri senin bana dediğine göre: 1100
   * 3. paramtere: maksimum değeri senin bana dediğine göre: 1900
   */
  ESC.attach(PIN,minESC,maxESC);
}
void loop() {
  //Joystickten gelen analog veriyi okuyor.
  //Analog pini hangisiyse "A0" olan yeri değiştir.
  joystickValue = analogRead(A0);
 
  //map yapıp servonun okuyacağı veriye dönüştürüyor.
  joystickValue = map(potValue, 0, 1023, 0, 180);
  //ESC'ye veriyi gönderiyor
  ESC.write(joystickValue);
}
