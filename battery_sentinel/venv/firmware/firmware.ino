#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ===== 핀 정의 =====
#define ONE_WIRE_BUS   4
#define PIN_PWM_H1    25   // MOSFET 1 - Heater H1
#define PIN_PWM_H2    26   // MOSFET 2 - Heater H2
// 릴레이는 아직 미연결

Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

int h1_pwm = 0;
int h2_pwm = 0;
unsigned long lastSend = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);  // SDA, SCL

  pinMode(PIN_PWM_H1, OUTPUT);
  pinMode(PIN_PWM_H2, OUTPUT);
  analogWrite(PIN_PWM_H1, 0);
  analogWrite(PIN_PWM_H2, 0);

  sensors.begin();
  if (!ina219.begin()) {
    Serial.println("{\"error\":\"INA219 not found\"}");
  }
}

void loop() {
  // PC에서 명령 수신: "CMD:h1,h2\n"  예) CMD:80,0
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("CMD:")) {
      String data = line.substring(4);
      int comma = data.indexOf(',');
      if (comma > 0) {
        h1_pwm = data.substring(0, comma).toInt();
        h2_pwm = data.substring(comma + 1).toInt();
        analogWrite(PIN_PWM_H1, constrain(h1_pwm, 0, 255));
        analogWrite(PIN_PWM_H2, constrain(h2_pwm, 0, 255));
      }
    }
  }

  // 1초마다 센서 데이터 전송
  if (millis() - lastSend >= 1000) {
    lastSend = millis();
    sensors.requestTemperatures();

    float t1 = sensors.getTempCByIndex(0);
    float t2 = sensors.getTempCByIndex(1);
    float t3 = sensors.getTempCByIndex(2);
    float t4 = sensors.getTempCByIndex(3);

    if (t1 < -50) t1 = 25.0;
    if (t2 < -50) t2 = 25.0;
    if (t3 < -50) t3 = 25.0;
    if (t4 < -50) t4 = 25.0;

    float v = ina219.getBusVoltage_V();
    float i_mA = ina219.getCurrent_mA();
    if (i_mA < 0) i_mA = 0;

    Serial.print("{\"T1\":"); Serial.print(t1, 2);
    Serial.print(",\"T2\":"); Serial.print(t2, 2);
    Serial.print(",\"T3\":"); Serial.print(t3, 2);
    Serial.print(",\"T4\":"); Serial.print(t4, 2);
    Serial.print(",\"V\":");  Serial.print(v, 2);
    Serial.print(",\"I\":");  Serial.print(i_mA, 1);
    Serial.print(",\"H1\":"); Serial.print(h1_pwm);
    Serial.print(",\"H2\":"); Serial.print(h2_pwm);
    Serial.println("}");
  }
}