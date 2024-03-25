#ifndef LOGGER_H
#define LOGGER_H


#include <string>
#include <unordered_map>

#include <petscsys.h>


namespace cppoqss {


class Logger
{
public:
    static Logger kLogger; 

    void add_stage(const std::string& stage_name);
    void push(const std::string& stage_name) const;
    void pop() const;

private:
    Logger() = default;

    std::unordered_map<std::string, PetscLogStage> log_stages_;
};


} // namespace cppoqss
   

#endif
