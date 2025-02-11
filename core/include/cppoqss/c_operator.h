#ifndef CPPOQSS_C_OPERATOR_H
#define CPPOQSS_C_OPERATOR_H


#include <cstddef>
#include <string>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_space.h>


namespace cppoqss {


/**
 * Must be implemented by user.
 */
template<class SourceInfoType, class TargetInfoType>
class IConstantCollapseImpl : public ICOperator
{
public:
    virtual ~IConstantCollapseImpl() { }

    virtual SourceInfoType get_source_info(const StateSpace& state_space, const MyIndexType sindex) const = 0;
    virtual TargetInfoType get_target_info(const StateSpace& state_space, const MyIndexType sindex) const = 0;
    virtual double get_constant_gamma_sum(const SourceInfoType& source_info) const = 0;
    virtual CollapseGamma get_constant_gamma(const SourceInfoType& source_info, const TargetInfoType& target_info) const = 0;
    virtual CollapseGamma get_constant_gamma_to_outer_state(const SourceInfoType& source_info, const std::string& outer_target_name) const = 0;

    bool is_constant() const final { return true; }

    double get_gamma_sum(const StateSpace& state_space, const MyIndexType source_sindex, const double t) const final;
    CollapseGamma get_gamma(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex, const double t) const final;
    bool is_gamma_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex) const final;
    CollapseGamma get_gamma_to_outer_state(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index, const double t) const final;
    bool is_gamma_to_outer_state_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index) const final;
};


template<class SourceInfoType, class TargetInfoType>
double IConstantCollapseImpl<SourceInfoType, TargetInfoType>::get_gamma_sum(const StateSpace& state_space, const MyIndexType source_sindex, const double t) const
{
    return this->get_constant_gamma_sum(this->get_source_info(state_space, source_sindex));
}


template<class SourceInfoType, class TargetInfoType>
CollapseGamma IConstantCollapseImpl<SourceInfoType, TargetInfoType>::get_gamma(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex, const double t) const
{
    return this->get_constant_gamma(this->get_source_info(state_space, source_sindex), this->get_target_info(state_space, target_sindex));
}


template<class SourceInfoType, class TargetInfoType>
bool IConstantCollapseImpl<SourceInfoType, TargetInfoType>::is_gamma_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex) const
{
    return this->get_constant_gamma(this->get_source_info(state_space, source_sindex), this->get_target_info(state_space, target_sindex)).gamma != 0.0;
}


template<class SourceInfoType, class TargetInfoType>
CollapseGamma IConstantCollapseImpl<SourceInfoType, TargetInfoType>::get_gamma_to_outer_state(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index, const double t) const
{
    return this->get_constant_gamma_to_outer_state(this->get_source_info(state_space, source_sindex), state_space.get_outer_state_name(outer_target_index));
}


template<class SourceInfoType, class TargetInfoType>
bool IConstantCollapseImpl<SourceInfoType, TargetInfoType>::is_gamma_to_outer_state_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index) const
{
    return this->get_constant_gamma_to_outer_state(this->get_source_info(state_space, source_sindex), state_space.get_outer_state_name(outer_target_index)).gamma != 0.0;
}


template<class SourceInfoType, class TargetInfoType>
class RuntimeCalculatingConstantCOperator : public IConstantCollapseImpl<SourceInfoType, TargetInfoType>
{ };


} // namespace cppoqss


#endif
