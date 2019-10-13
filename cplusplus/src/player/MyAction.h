/**
 * A type/Struct to hold action ID + extra information.
 * 
 **/ 

#include "../util/Position.h"

struct MyAction {
    int actionID;
    Position pos;
    
    MyAction(){};
    MyAction(int actID, Position p) : actionID(-1),pos(-1,-1) {
        actionID = actID;
        pos = p;
    }
};
