#include <cppoqss/logger.h>

#include <petscsys.h>


namespace cppoqss {


Logger Logger::kLogger;


void Logger::add_stage(const std::string& stage_name)
{
    if (log_stages_.find(stage_name) == log_stages_.end()) {
	PetscLogStage stage;
	PetscLogStageRegister(stage_name.c_str(), &stage);
	log_stages_.emplace(stage_name, stage);
    }
}


void Logger::push(const std::string& stage_name) const
{
    PetscLogStagePush(log_stages_.at(stage_name));
}


void Logger::pop() const
{
    PetscLogStagePop();
}


} // namespace cppoqss
