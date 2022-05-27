#include <iostream>
using namespace std;
class Fruit {
    public:
    Fruit() {
        cout<<"Fruit"<<endl;
    }
    virtual ~Fruit() // need virtual to prevent memory leak
    {
        cout<<"Fruit destructor"<<endl;
    }
};

class Apple : public Fruit {
    public:
    Apple() {
        cout<<"Apple"<<endl;
    }
    ~Apple()
    {
        cout<<"Apple destructor"<<endl;
    }
};

int main() {
    cout << "Hello World!";
    Fruit* f= new Apple();
    delete f;
    return 0;
}