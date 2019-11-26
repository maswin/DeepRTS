//
// Created by Per-Arne on 24.02.2017.
//

#include "Unit.h"
#include "../player/Player.h"
#include "../Game.h"
#include "../util/Pathfinder.h"
#include "./UnitManager.h"
#include <random>
#include <climits>

Unit::Unit(Player &player):
        game(&player.getGame()),
        config(player.config),
        player_(player),
        stateManager(&player.getGame().stateManager)
{

    // Harvesting
    // Modifying harvest interval so that everything happens per tick
    harvestInterval = 1 * config.tickModifier;
    combatInterval = 1 * config.tickModifier;
    walking_interval = 1 * config.tickModifier;
    id = player.getGame().units.size();
    state = stateManager->despawnedState;
    stateList.reserve(50);
}
void Unit::spawn(Tile &_toSpawnOn, int initValue) {
    spawnTimer = initValue;
    spawnTileID = _toSpawnOn.id;
    player_.food += foodProduction;
    player_.foodConsumption += foodConsumption;
    transitionState(stateManager->spawnState);
    enqueueState(stateManager->idleState);
}

void Unit::moveRelative(int x, int y) {


    int newX = tile->x + x;
    int newY = tile->y + y;

    Tile &moveTile = player_.getGame().tilemap.getTile(newX, newY);

    move(moveTile);
}

void Unit::rightClickRelative(int x, int y) {
    if (!tile) return; // Not standing on any tiles

    int newX = tile->x + x;
    int newY = tile->y + y;

    Tile &clickTile = player_.getGame().tilemap.getTile(newX, newY);
    rightClick(clickTile);
}

void Unit::move(Tile &targetTile){
    // If unit cannot move at all (Building for example)
    if (!canMove){
        return;
    }

    // If unit is trying to move to the same tile
    if (targetTile.id == tile->id){
        return;
    }

    this->walkingGoalID = targetTile.id;
    transitionState(stateManager->walkingState);
}


void Unit::setPosition(Tile &newTile) {

    if(tile){                       // If unit already stands on a tile
        clearTiles();
    }

    for(auto &t : player_.getGame().tilemap.getTileArea(newTile, width, height)) {
        setStateForTile(t);
        t->setOccupant(this);
    }


    newTile.setOccupant(this);         // Set occupant of new tile to this
    tile = &newTile;                     // Set this units tile to new tile
    Position newPos = tile->getPosition();
    setDirection(newPos.x, newPos.y);
    worldPosition = newPos;
}

void Unit::update() {
    state->update(*this);
}

Tile *Unit::centerTile() {
    int addX = floor(width / 2);
    int addY = floor(height / 2);

    if (addX == 0 && addY == 0) {
        assert(tile);
        return tile;
    }

    return &player_.getGame().tilemap.getTile(tile->x + addX, tile->y + addY);
}

bool Unit::build(int idx) {
    //if(state->id != Constants::State::Idle)
    if(state->id != Constants::State::Idle)
        return false;

    if((idx < 0 || idx >= buildInventory.size()))
        return false;

    Unit newUnit = UnitManager::constructUnit(buildInventory[idx], player_);


    // Check food restriction
//    if(config.foodLimit && newUnit.foodConsumption + player_.foodConsumption > player_.food) {
//        return false;
//    }


    // PlacementTile is based on dimension of the new unit. For example; town hall has
    // 3x Width && 3x Height. We then want to place  the building by the middle tile;

    int x = tile->x - floor(newUnit.width/2);
    int y = tile->y - floor(newUnit.height/2);

    if(!position_in_bounds(x, y)) {
        return false;
    }

    Tile &placementTile = player_.getGame().tilemap.getTile(x, y);
    if(!player_.canAfford(newUnit)) {
        //std::cout << "Cannot afford " << newUnit->name << std::endl;
        return false;
    }



    if(player_.canPlace(*this, newUnit, placementTile)) {

        Unit &unit = player_.addUnit(newUnit.typeId);
        //unit.player_ = player_;
        unit.builtByID = this->id;

        if(!structure && unit.structure) {
            // *this is a unit (peasant), which builds a building
//            despawn();

            buildEntityID = unit.id; // Set id of buildEntity
            transitionState(stateManager->buildingState);
            unit.spawn(placementTile, 0);     // Spawn build entity
            unit.setPosition(placementTile);  // Set position of build entity


        }else if(structure && !unit.structure){
            // Structure builds unit (Ie: TownHall builds Peasant)
            buildEntityID = unit.id;    // Set build entity ID
            // build entity has no tile, spawn on nearest walkable
            Tile *firstWalkable = Pathfinder::find_first_walkable_tile(centerTile());
            assert(firstWalkable);
            unit.spawn(*firstWalkable, 0);  // Spawn build entity

            transitionState(stateManager->buildingState);
        }


//        player_.removeGold(unit.goldCost);
//        player_.removeLumber(unit.lumberCost);
//        player_.removeOil(unit.oilCost);
        player_.sUnitsCreated += 1;

        return true;

    } else {
        return false;
    }

}

void Unit::despawn() {

    for(auto &p: player_.getGame().players) {
        if(p.getTargetedUnitID() == this->id)
            p.setTargetedUnitID(-1);
    }

    player_.food -= foodProduction;
    player_.foodConsumption -= foodConsumption;
    clearTiles();
    transitionState(stateManager->despawnedState);
}

void Unit::clearTiles(){
    for(auto &t : player_.getGame().tilemap.getTileArea(*tile, width, height)) {
        clearStateForTile(t);
        t->setOccupant(NULL);
    }
    tile->setOccupant(NULL);
    tile = NULL;
}

void Unit::rightClick(Tile &targetTile) {
    // For these actions, the unit must be spawned already
    if(!tile) {
        return;
    }


    stateList.clear();
    transitionState();


    if(targetTile.isHarvestable()){
        //  Harvest
        harvest(targetTile);
    }
    else if(targetTile.isAttackable(*this)){
        // Attack
        attack(targetTile);
    }
    /*else if(targetTile.isRepairable(*this)){
        repair(targetTile);
    }*/
    else {
        // Walk
        move(targetTile);
    }

}

void Unit::attack(Tile &tile) {
    if(!canAttack)
        return;

    Unit* target = tile.getOccupant();

    // Target may have died from another same tick
    if (!target) {
        transitionState();
        return;
    }

    combatTargetID = target->id;
    transitionState(stateManager->combatState);




}

void Unit::harvest(Tile &tile) {
    if(!canHarvest)
        return;

    this->harvestTargetID = tile.id;
    transitionState(stateManager->harvestingState);

}

void Unit::enqueueState(std::shared_ptr<BaseState> state) {
    stateList.push_back(state->id);
}

void Unit::transitionState() {
    if(stateList.empty()) {
        // No states to transition to, enter idle state
        std::shared_ptr<BaseState> nextState = stateManager->idleState;
        //std::cout << id<< " State Transition: " << state->name << " ==> " << nextState->name << std::endl;
        state->end(*this);
        state = nextState;
        state->init(*this);
        return;
    }

    auto nextStateId = stateList.back();


    state->end(*this);
    auto nextState = stateManager->getByID(nextStateId);
    //std::cout << id<< " State Transition: " << state->name << " ==> " << nextState->name << std::endl;
    state = nextState;
    stateList.pop_back();
    //state->init(*this);


}

void Unit::transitionState(std::shared_ptr<BaseState> nextState) {
    //std::cout << id << " State Transition: " << state->name << " ==> " << nextState->name  << std::endl;


    state->end(*this);
    state = std::move(nextState);
    state->init(*this);
}

int Unit::distance(Tile &target) {
    int dim = 0; // TODO
    double d = hypot(tile->x - (target.x + dim), tile->y - (target.y + dim));
    return (int)d - dim;

}


int Unit::distance(Unit & target) {
    int targ_x = target.tile->x;
    int targ_y = target.tile->y;
    int dim_x = target.width / 2;
    int dim_y = target.height / 2;

    int my_x = tile->x;
    int my_y = tile->y;

    int closest_x = 0;
    if(std::abs(targ_x - my_x) > std::abs(targ_x + dim_x - my_x)) {
        closest_x = targ_x + dim_x;
    } else {
        closest_x = targ_x;
    }

    int closest_y = 0;
    if(std::abs(targ_y - my_y) > std::abs(targ_y + dim_y - my_y)) {
        closest_y = targ_y + dim_y;
    } else {
        closest_y = targ_y;
    }

    double d = hypot(my_x - closest_x, my_y - closest_y);

    return (int)d;

}

Position Unit::distanceVector(Tile &target){
    int dx = tile->x - target.x;
    int dy = tile->y - target.y;

    return {dx, dy};
}

Unit* Unit::closestRecallBuilding() {
    Unit* closest = NULL;
    int dist = INT_MAX;
    for(auto &unit : player_.getGame().units) {
        if(unit.recallable && unit.player_.getId() == player_.getId() && unit.tile) {
            int d = distance(*unit.tile);
            if(d < dist) {
                dist = d;
                closest = &unit;
            }
        }
    }
    return closest;
}

void Unit::afflictDamage(int dmg_) {
    health = std::max(0, health - dmg_);

    if (health <= 0) {
        transitionState(stateManager->deadState);
    }

}

bool Unit::isDead() {
    return state->id == Constants::State::Dead || state->id == Constants::State::Despawned;
}

int Unit::getDamage(Unit &target) {

    // TODO better random
    double output = damageMin + (rand() % (damageMax - damageMin + 1));
    double myDamage = (output - (target.armor*0.12)) + damagePiercing;
    myDamage = std::max(0.0, myDamage);

    double mini = myDamage * .50;
    double output2 = mini + (rand() % (int)(myDamage - mini + 1));

    return floor(output2);

}

void Unit::setDirection(Position &dir){
    setDirection(dir.x, dir.y);
}

Tile *Unit::getNextTile(){
    if(walking_path.empty()) {
        return NULL;
    }
    return walking_path.back();
}

void Unit::setDirection(int newX, int newY){
    int oldX = worldPosition.x;
    int oldY = worldPosition.y;

    int dx = (newX - oldX);
    int dy = (newY - oldY);

    if (dx > 0 && dy > 0) {
        // Down Right
        direction = Constants::Direction::DownRight;
        //std::cout << "Down Right" << std::endl;
    } else if (dx < 0 && dy > 0) {
        // Down Left
        direction = Constants::Direction::DownLeft;
        //std::cout << "Down Left" << std::endl;
    } else if (dx > 0 && dy < 0) {
        // Up Right
        direction = Constants::Direction::UpRight;
        //std::cout << "Up Right" << std::endl;
    } else if (dx < 0 && dy < 0) {
        // Up Left
        direction = Constants::Direction::UpLeft;
        //std::cout << "Up Left" << std::endl;
    } else if (dx > 0 && dy == 0) {
        // Right
        direction = Constants::Direction::Right;
        //std::cout << "Right" << std::endl;
    } else if (dx < 0 && dy == 0) {
        // Left
        direction = Constants::Direction::Left;
        //std::cout << "Left" << std::endl;
    } else if (dx == 0 && dy < 0) {
        // Up
        direction = Constants::Direction::Up;
        //std::cout << "Up" << std::endl;
    } else if (dx == 0 && dy > 0) {
        // Down
        direction = Constants::Direction::Down;
        //std::cout << "Down" << std::endl;
    }



}

bool Unit::operator==(int otherID) const
{
    return otherID == id;
}

void Unit::tryAttack()
{
    if (!tile) {
        // FAIL
        return;
    }

    std::vector<Tile *> availableAttackable = player_.getGame().tilemap.neighbors(*tile, Constants::Pathfinding::Attackable);
    if (availableAttackable.empty()) {
        // Fail
        return;
    }else {
        // Success
        attack(*availableAttackable.back());
    }

}

void Unit::tryMyAttack(Position p)
{
    Tile tile = player_.getGame().tilemap.getTile(p.x,p.y);
    if (! &tile) {
        // FAIL
        return;
    }

    // Clear if any previous state is in queue (i.e waiting to attack)
    stateList.clear();
    attack(tile);
}

void Unit::tryMove(int16_t x, int16_t y)
{
    if (!tile) {
        // FAil
        return;
    }

    int newX = tile->x + x;
    int newY = tile->y + y;

    if(!position_in_bounds(newX, newY)) {
        return;
    }

    Tile &tile = player_.getGame().tilemap.getTile(newX, newY);

    if (tile.isWalkable()) {
        move(tile);
        return;

    }

    // Allow to automatically attack if config has enabled this
    if(config.autoAttack && tile.isAttackable(*this)) {
        attack(tile);
        return;
    }


    if(config.harvestForever && tile.isHarvestable()) {
        harvest(tile);
        return;
    }

    // Failed, Cannot move
}

void Unit::tryMyMove(Position pos) {
    if (!tile) {
        // FAil
        return;
    }

    if(!position_in_bounds(pos.x, pos.y)) {
        return;
    }

    Tile &tile = player_.getGame().tilemap.getTile(pos.x, pos.y);

    if (tile.isWalkable()) {
        move(tile);
        return;
    }

    // Allow to automatically attack if config has enabled this
    if(config.autoAttack && tile.isAttackable(*this)) {
        attack(tile);
        return;
    }

    if(config.harvestForever && tile.isHarvestable()) {
        harvest(tile);
        return;
    }

}
void Unit::tryHarvest()
{
    if (!tile) {
        // FAIL
        return;
    }

    std::vector<Tile *> availableHarvestable = player_.getGame().tilemap.neighbors(*tile, Constants::Pathfinding::Harvestable);
    if (availableHarvestable.empty()) {
        // Fail
        return;
    }
    else {
        harvest(*availableHarvestable.back());
    }
}

void Unit::tryMyHarvest(Position p)
{   
    Tile tile = player_.getGame().tilemap.getTile(p.x,p.y);
    if (! &tile) {
        // FAIL
        return;
    }

    // Clear if any previous state is in queue (i.e waiting to attack)
    stateList.clear();

    harvest(tile);
}


Tile &Unit::getSpawnTile() {
    assert(spawnTileID != -1);
    return player_.getGame().tilemap.getTiles()[spawnTileID];
}

Tile *Unit::getTile(int tileID) {
    if(tileID == -1) {
        return NULL;
    }
    return &player_.getGame().tilemap.getTiles()[tileID];
}

Unit &Unit::getBuiltBy() {
    assert(builtByID != -1);
    return player_.getGame().units[builtByID];
}

Unit &Unit::getBuildEntity() {
    assert(buildEntityID != -1);
    return player_.getGame().units[buildEntityID];
}

Unit *Unit::getCombatTarget() {
    //assert(combatTargetID != -1);
    if (combatTargetID == -1) {
        return NULL;
    }
    return &player_.getGame().units[combatTargetID];
}

std::set<int> Unit::getVisionTileIDs() {

    std::set<int> tileIDs = std::set<int>();
    if(!tile){
        return tileIDs;
    }

    // Current tile is the upper right
    int tileX = tile->x;
    int tileY = tile->y;

    // Calculate vision tiles
    for(auto x = -sight; x < width + sight; x++) {
        for(auto y = -sight; y < height + sight; y++){
            int tX = tileX + x;
            int yY = tileY + y;
            int idx = player_.getGame().map.MAP_HEIGHT*yY + tX;
            tileIDs.insert(idx);
        }
    }
    return tileIDs;
}

bool Unit::position_in_bounds(int x, int y) {
    if(
            x < 0 || x > this->player_.getGame().map.MAP_WIDTH ||
            y < 0 || y > this->player_.getGame().map.MAP_HEIGHT
            ){
        return false;
    }
    return true;
}

Player &Unit::getPlayer() {
    return player_;
}

void Unit::clearStateForTile(Tile *t){
    player_.getGame().state(t->x, t->y, 1) = 0; // Player ID
    player_.getGame().state(t->x, t->y, 2) = 0; // 1 if its a building
    player_.getGame().state(t->x, t->y, 3) = 0; // 1 if its a unit
    player_.getGame().state(t->x, t->y, 4) = 0; // Unit Type
    player_.getGame().state(t->x, t->y, 5) = 0; // Unit Health percent
    player_.getGame().state(t->x, t->y, 6) = 0; // Unit Unit State
    player_.getGame().state(t->x, t->y, 7) = 0; // Unit Total Carry
    player_.getGame().state(t->x, t->y, 8) = 0; // Unit Attack Score
    player_.getGame().state(t->x, t->y, 9) = 0; // Unit Defense Score
}
void Unit::setStateForTile(Tile *t){
    player_.getGame().state(t->x, t->y, 1) = player_.getId(); // Player ID
    player_.getGame().state(t->x, t->y, 2) = (canMove) ? 0 : 1; // 1 if its a building
    player_.getGame().state(t->x, t->y, 3) = (canMove) ? 1 : 0; // 1 if its a unit
    player_.getGame().state(t->x, t->y, 4) = int(typeId); // Unit Type
    player_.getGame().state(t->x, t->y, 5) = health / health_max; // Unit Health percent
    player_.getGame().state(t->x, t->y, 6) = (int)state->id; // Unit Unit State
    player_.getGame().state(t->x, t->y, 7) = oilCarry + goldCarry + lumberCarry; // Unit Total Carry
    player_.getGame().state(t->x, t->y, 8) = damageMin + damageMax + damagePiercing; // Unit Attack Score
    player_.getGame().state(t->x, t->y, 9) = armor; // Unit Defense Score
}