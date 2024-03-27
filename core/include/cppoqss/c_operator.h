#ifndef CPPOQSS_C_OPERATOR_H
#define CPPOQSS_C_OPERATOR_H


#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <boost/container_hash/hash.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/unordered_map.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/mpi_helper.h>
#include <cppoqss/progress_bar.h>
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

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);
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
template<class Archive>
void IConstantCollapseImpl<SourceInfoType, TargetInfoType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<ICOperator>(*this);
}


/**
 * The function hash_key() must be callable on SourceInfoType, ToRhoCacheKeyType, and ToPhiCacheKeyType.
 */
template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
class CachedConstantCOperator : public IConstantCollapseImpl<SourceInfoType, TargetInfoType>
{
public:
    virtual ToRhoCacheKeyType get_to_rho_cache_key(const SourceInfoType& source_info, const TargetInfoType& target_info) const = 0;
    virtual ToPhiCacheKeyType get_to_phi_cache_key(const SourceInfoType& source_info) const = 0;
    virtual CollapseGamma calculate_constant_gamma(const SourceInfoType& source_info, const TargetInfoType& target_info) const = 0;
    virtual CollapseGamma calculate_constant_gamma_to_outer_state(const SourceInfoType& source_info, const std::string& outer_target_name) const = 0;


    void initialize(const StateSpace& state_space, const MyIndexType source_sindex_start, const MyIndexType source_sindex_end) final;

    double get_constant_gamma_sum(const SourceInfoType& source_info) const final;
    CollapseGamma get_constant_gamma(const SourceInfoType& source_info, const TargetInfoType& target_info) const final;
    virtual CollapseGamma get_constant_gamma_to_outer_state(const SourceInfoType& source_info, const std::string& outer_target_name) const final;

private:
    friend class boost::serialization::access;

    struct ToPhiCacheTrueKeyType
    {
	friend std::size_t hash_value(const ToPhiCacheTrueKeyType& v)
	{
	    std::size_t seed = 0;

	    boost::hash_combine(seed, v.source_key);
	    boost::hash_combine(seed, v.target_key);

	    return seed;
	}

	bool operator==(const ToPhiCacheTrueKeyType& other) const;

	ToPhiCacheKeyType source_key;
	std::string target_key;
    };

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);

    boost::unordered_map<ToRhoCacheKeyType, CollapseGamma> to_rho_cache_;
    boost::unordered_map<ToPhiCacheTrueKeyType, CollapseGamma> to_phi_cache_;
    boost::unordered_map<SourceInfoType, double> sum_cache_;
};


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
void CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::initialize(const StateSpace& state_space, const MyIndexType source_sindex_start, const MyIndexType source_sindex_end)
{
    const MyIndexType n_rho_target = state_space.get_n_dim_rho();

    const MyIndexType n_phi_target = state_space.get_n_dim_phi();
    
    ProgressBar::InitBar("cache c-operator", source_sindex_end - source_sindex_start);

    for (MyIndexType source_sindex = source_sindex_start; source_sindex < source_sindex_end; ++source_sindex) {
	const SourceInfoType source_info = this->get_source_info(state_space, source_sindex);

	double sum = 0;

	for (MyIndexType target_sindex = 0; target_sindex < n_rho_target; ++target_sindex) {
	    const TargetInfoType target_info = this->get_target_info(state_space, target_sindex);

	    CollapseGamma gamma = calculate_constant_gamma(source_info, target_info);

	    if (gamma.gamma != 0.0) {
		to_rho_cache_.emplace(get_to_rho_cache_key(source_info, target_info), gamma);
		sum += gamma.get_collapse_gamma();
	    }
	}

	for (MyIndexType outer_target_index = 0; outer_target_index < n_phi_target; ++outer_target_index) {
	    const std::string outer_target_name = state_space.get_outer_state_name(outer_target_index);

	    ToPhiCacheTrueKeyType key { get_to_phi_cache_key(source_info), outer_target_name };

	    CollapseGamma gamma = calculate_constant_gamma_to_outer_state(source_info, outer_target_name); 

	    if (gamma.gamma != 0.0) {
		to_phi_cache_.emplace(key, gamma);
		sum += gamma.get_collapse_gamma();
	    }
	}

	sum_cache_.emplace(source_info, sum);

	ProgressBar::ProgressStep();
    }

    const size_t n_collapse_mode = to_rho_cache_.size() + to_phi_cache_.size();
    const size_t n_collapse_mode_sum = boost::mpi::all_reduce(mpi_helper::world, n_collapse_mode, std::plus<size_t>());

    if (mpi_helper::is_printing_rank()) {
	printf("cached operator initialized\n");
	printf("number of unique and cached collapse mode: %zu\n", n_collapse_mode_sum);
    }
}


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
double CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::get_constant_gamma_sum(const SourceInfoType& source_info) const
{
    auto found = sum_cache_.find(source_info);

    if (found != sum_cache_.end()) return found->second;
    else return 0.0;
}


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
CollapseGamma CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::get_constant_gamma(const SourceInfoType& source_info, const TargetInfoType& target_info) const
{
    auto found = to_rho_cache_.find(get_to_rho_cache_key(source_info, target_info));

    if (found != to_rho_cache_.end()) return found->second;
    else return CollapseGamma(0.0);
}


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
CollapseGamma CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::get_constant_gamma_to_outer_state(const SourceInfoType& source_info, const std::string& outer_target_name) const
{
    auto found = to_phi_cache_.find(ToPhiCacheTrueKeyType { get_to_phi_cache_key(source_info), outer_target_name });

    if (found != to_phi_cache_.end()) return found->second;
    else return CollapseGamma(0.0);
}


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
bool CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::ToPhiCacheTrueKeyType::operator==(const typename CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::ToPhiCacheTrueKeyType& other) const
{
    return 
	source_key == other.source_key
	&& target_key == other.target_key
	;
}


template<class SourceInfoType, class TargetInfoType, class ToRhoCacheKeyType, class ToPhiCacheKeyType>
template<class Archive>
void CachedConstantCOperator<SourceInfoType, TargetInfoType, ToRhoCacheKeyType, ToPhiCacheKeyType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IConstantCollapseImpl<SourceInfoType, TargetInfoType>>(*this);
}


template<class SourceInfoType, class TargetInfoType>
class RuntimeCalculatingConstantCOperator : public IConstantCollapseImpl<SourceInfoType, TargetInfoType>
{
public:
    void initialize(const StateSpace& state_space, const MyIndexType source_sindex_start, const MyIndexType source_sindex_end) override { }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);
};


template<class SourceInfoType, class TargetInfoType>
template<class Archive>
void RuntimeCalculatingConstantCOperator<SourceInfoType, TargetInfoType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IConstantCollapseImpl<SourceInfoType, TargetInfoType>>(*this);
}


} // namespace cppoqss


#endif
