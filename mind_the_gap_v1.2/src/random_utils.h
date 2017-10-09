#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>


namespace rd{
// shuffle
// generate random int
// generate random float uniform
// generate random float normal

struct Random{
    static const int SEED;
    static std::default_random_engine re;
    static std::mt19937 gen;
    static std::uniform_real_distribution<> uniform_01;
};

double random();
}

#endif // RANDOM_UTILS_H
