#include <iostream>
#include <unistd.h>
#include <ctime>
double percent = 0.00;
long chanse = 0; //если меньше 3 то повезло повезло
bool run = true; //для депука ведь наин хуета ленивая

int main() {
  std::cout << "Превращаем тебя в негра" << std::endl;
  srand(time(0));
  chanse = rand() % 50;
  if(run) {
  sleep(1);
  for(int i = 1; i < 4; i++) {
    std::cout << i << std::endl;
    sleep(1);
  }
  while(percent < 80) { 
    usleep(50000);
    percent = percent + 0.5;
    std::cout << percent << "%" << std::endl;
  }
  while(percent < 99) {
    usleep(20000);
    percent = percent + 0.01;
    std::cout << percent << "%" << std::endl;
  }
  while(percent < 100) {
    usleep(20000);
    percent = percent + 0.001;
    std::cout << percent << "%" << std::endl;
  }
  sleep(2);
  }
  
  if(chanse > 3) {
  std::cout << "АЩИПКА!11!!" << std::endl;
  std::cout << "STOP 0x0000010100101010бфыбыбфбфывб" << std::endl;
  std::cout << "крч заебал давай по новой" << std::endl;
  }else {
  std::cout << "еее ты фурри" << std::endl;
  }
}
