# Motion tracking and ML for rehabulation
# Arduino uno code
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"

MPU6050 mpu(0x68);

int16_t ax, ay, az;
int16_t gx, gy, gz;

int data, sendData;

void setup()
{
  Wire.begin();
  Serial.begin(9600);
  mpu.initialize();
}

void loop()
{
  sendData = 0;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  data = map(ax, -11500, 11500, 0, 255);

  for(int i=0; i<10; i++)
  {
    sendData = sendData + data;
    delay(5);
  }
  sendData = sendData / 10;

  if(sendData >= 255)
      sendData = 255;
  else if(sendData <= 0)
      sendData = 0;

  Serial.write(sendData);
  delay(10);
}

# game integration 
import processing.serial.*;

Serial port;

 int data;
int movby;
float x=300;
float y=00;
float spdy=6;
float spdx=0;
float posxr,posxl,posyr,posyl,ballpos;
int score = 0, high = 0, high1 = 0;
float speed=0,speed1;
float flag=0;
float ballsize=20;
float temp;
int z = 0;
boolean state = true;

void setup()
{
  size(600,600,P2D);
  smooth();
  port = new Serial(this,Serial.list()[0],9600);
  println(port);
  println(Serial.list());
}




void draw()
{
   if (port.available()>0 && port.available()<30)
  {
    data=port.read();
    movby = int(map(float(data), 255,0,30,570));
    println(movby);
  }


  if (flag==0)
    strtscrn();
 if (flag==1)
    game();
 if (flag==3)
    lastscrn();

 

 fill(18,1,8);
 textSize(20);
 text("Yahiya Mulla", 260, 590);

 textSize(22);
 text("Score:",10,20);
 text(score, 80, 20);
 fill(0, 102, 153);

 fill(18,1,8);
 textSize(22);
 text("High Score:",420,20);
 text(high, 550, 20);

}


void strtscrn()
{
  score = 0;
  high1 = 0;
  cursor();
  background(255); //Color of the backgroud
  fill(18,1,8);
  textSize(52);
  text("WELCOME", 200,300);
  fill(18,100,80);
  textSize(32);
  text("Select the speed", 200,333);
  fill(180,100,8);
  textSize(22);
  text("1   2    3   4", 255,370);
  //println(mouseX, mouseY);
  if (mouseY > 340 && mouseY < 380)  
  {
    cursor(HAND);
    if(mousePressed==true && flag==0)
    {
      if (mouseX >245 && mouseX <275)
      {
        speed=1;
      }
      if (mouseX >280 && mouseX <315)
      {
        speed=2;
      }
      if (mouseX >322 && mouseX <355)
      {
        speed=3;
      }
      if (mouseX >357 && mouseX <387)
      {
        speed=4;
      }
    flag=1;
    speed1 = speed;
    }
  }
}


void game()
{
  if(score == 0)
      speed = speed1;

  high1 = 0;
  if((z > 0) && (state == true))
  {
    score = 0;
    state = false;
  }
  noCursor();
  background(255); //Color of the backgroud
  y=y+spdy; //speed and positon of ball in Y axis
  x=x+spdx; //speed and positon of ball in X axis 

  rectMode(CENTER);  
  fill(16,22,162);
  rect(movby,530,140,13); //The plate

  posxr=movby+80;
  posxl=movby-80;
  posyr=530+15;
  posyl=530-13;

  if (( (posyl < y) && (y < posyr) ) && ( (posxl < x) && (x < posxr) ))  //Plate and ball meeting
  {
    spdy=-(speed*3);
  

    if (x<movby) //Pad left deflection
    {
      ballpos=movby-x;
      spdx=-(ballpos/5);
   // println(ballpos);
    }

    if (x>movby)  //Pad right deflection
    {
       ballpos=x-movby;
       spdx=+(ballpos/5);
      //println(ballpos);
    }

  }
  
  if (x<=0 ) // Left margin deflection
  spdx=(speed*3);


  if (y<=0 ) // Top margin deflection
  {
    spdy=(speed*3);
    score=score+1;
    if(high < score)
    {
      high = score;
    }
  }

  if (x>=600) // Right margin deflection
  spdx=-(speed*3);

  if (y>=600) 
  {
    background(250,0,0);//red background
    x=300;
    y=0;
    flag=3;
    //speed=1;
  }

    fill(88,250,68);
    ellipse(x,y,ballsize,20); //The ball


  if(score>=15 && score<=20)
  {
    ellipse(random(600),random(600),20,20);
    ellipse(random(600),random(600),20,20);
    ellipse(random(600),y,20,20);
    ellipse(x,random(600),20,20);///The ball
  }

  temp=x;
  high1 = score;
  
  if(score == 20)
  {
    if(speed == 1)
    {
      speed = speed +1;
    } 
    else if(speed == 2)
    {
      speed = speed +1;
    } 
     else if(speed == 3)
    {
      speed = speed +1;
    } 
  }
}


void lastscrn()
{
  
  background(250,0,0);
  textSize(32);
  text("GAME OVER", 200,300);
  fill(18,2,8);
  textSize(22);
  text("Your Score:", 200,335);
  text(score, 320,335);
  fill(80,2,8);
  text("Click here to try again..", 200,370);
  textSize(22);
  fill(80,2,8);
  text("Press S to select the speed", 200,400);
  
  if(high1 >= high)
  {
    textSize(40);
    fill(80,2,8);
    text("High Score....", 200,200);
  }
  cursor(HAND);
  if(mousePressed == true)
  {
    flag = 1;
    z = z + 1;
    state = true;
  }
  if(keyPressed)
  {
    switch(key)
    {
      case 's':
          flag = 0;
          break;
      case 'S':
          flag = 0;
          break;
    }
    
  }
  
}
