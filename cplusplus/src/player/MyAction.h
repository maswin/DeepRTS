/**
 * A type/Struct to hold action ID + extra information.
 * 
 **/ 

#include "../util/Position.h"

struct MyAction {
    int actionID;
    Position pos;
    MyAction(int actID, Position p) : pos(-1,-1) {
        actionID = actID;
        pos = p;
    }
};
