#include <iostream>
#include <unistd.h>
#include <pwd.h>
using namespace std;

int main() {
 struct passwd *pw = getpwuid(getuid());
 cout << "Microsoft Windows [Version 6.2.8102]" << endl; 
 cout << "(c) 2011 Microsoft Corporation. All rights reserved." << endl << endl;
 char cwd[1024];
 string user = getcwd(cwd,sizeof(cwd));
 string input = "";
 cout << "C:" << user << ">";
 
 while(1) {
 cin >> input;
 int returnCode = system(input.c_str());
 if (returnCode == 0) {
    }
    else {
        cout << "'" << input << "'" << " is not recognized as an internal or external command, operable program or batch file." << endl << endl;
    }
 cout << "C:" << user << ">";
}
 
}
