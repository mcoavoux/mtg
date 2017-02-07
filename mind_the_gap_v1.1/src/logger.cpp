#include "logger.h"



Logger::Logger():btime(0.0), total_time(0.0){}

void Logger::start(){
    btime = get_time();
}

void Logger::stop(){
    double end = get_time();
    double ctime = end - btime;
    total_time += ctime;
}

double Logger::get_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

double Logger::get_total_time(){
    return total_time;
}
