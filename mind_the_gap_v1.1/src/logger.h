#ifndef LOGGER_H
#define LOGGER_H

#include <vector>
#include <fstream>
#include <sys/time.h>

class Logger{

    double btime;
    double total_time;
public:

    Logger();
    void start();
    void stop();

    double get_time();

    double get_total_time();
};


#endif // LOGGER_H
