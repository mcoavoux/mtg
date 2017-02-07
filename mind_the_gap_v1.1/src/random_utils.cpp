
#include "random_utils.h"

namespace rd{

const int Random::SEED{1};
std::default_random_engine Random::re(SEED);
std::mt19937 Random::gen(re());
std::uniform_real_distribution<> Random::uniform_01(0.0, 1.0);


double random(){
    return Random::uniform_01(Random::gen);
}


}

