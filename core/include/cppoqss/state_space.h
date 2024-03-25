#ifndef STATE_SPACE_H
#define STATE_SPACE_H


#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <cppoqss/arithmetic.h>


namespace cppoqss {


class ISingleStateSpace
{
public:
    virtual ~ISingleStateSpace() {}

    virtual std::string get_space_name() const = 0;
    virtual MyIndexType get_n_dim() const = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) { }
};


template<class EigenValueType>
class SingleStateSpace : public ISingleStateSpace
{
public:
    template<class EigenValuesType>
    SingleStateSpace(const std::string& space_name, EigenValuesType eigen_values);

    const EigenValueType& get_eigen_value(const MyIndexType index) const { return *eigen_values_.at(index); }
    std::string get_unique_eigen_state_label(const MyIndexType index) const { return eigen_values_[index].get_unique_label(); }

    std::string get_space_name() const override { return space_name_; }
    MyIndexType get_n_dim() const override { return n_dim_; }

    void loop_over_states(std::function<void(const MyIndexType, const EigenValueType&)> process) const;

private:
    friend class boost::serialization::access;

    template<class Archive>
    friend void save_construct_data(Archive& ar, const SingleStateSpace<EigenValueType>* t, const unsigned int version)
    {
	ar << t->space_name_;
	ar << t->eigen_values_;
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);

    std::string space_name_;
    std::vector<std::unique_ptr<EigenValueType>> eigen_values_;
    MyIndexType n_dim_;
};


template<class EigenValueType>
template<class EigenValuesType>
SingleStateSpace<EigenValueType>::SingleStateSpace(const std::string& space_name, EigenValuesType eigen_values)
    : space_name_(space_name), eigen_values_(std::forward<EigenValuesType>(eigen_values)), n_dim_(eigen_values_.size()) 
{ }


template<class EigenValueType>
void SingleStateSpace<EigenValueType>::loop_over_states(std::function<void(const MyIndexType, const EigenValueType&)> process) const
{
    for (MyIndexType i = 0; i < n_dim_; ++i) {
	process(i, *eigen_values_[i]);
    }
}


template<class EigenValueType>
template<class Archive>
void SingleStateSpace<EigenValueType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<ISingleStateSpace>(*this);
    ar & space_name_;
    ar & eigen_values_;
    ar & n_dim_;
}


class StateSpace;


class IOperator
{
public:
    virtual ~IOperator() { }

    virtual void set_time(const double t) const = 0;
    virtual MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType i, const MyIndexType j, const double t) const = 0;
    virtual bool is_element_nonzero(const StateSpace& state_space, const MyIndexType i, const MyIndexType j) const = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) { }
};


struct CollapseGamma
{
    CollapseGamma(const double gamma, const bool ignore_decay=false, const bool ignore_filling=false)
	: gamma(gamma), ignore_decay(ignore_decay), ignore_filling(ignore_filling) { }

    double get_collapse_gamma() const { return ignore_decay ? 0.0 : gamma; }
    double get_fill_gamma() const { return ignore_filling ? 0.0 : gamma; }

    double gamma;
    bool ignore_decay;
    bool ignore_filling;
};


class ICOperator
{
public:
    virtual ~ICOperator() {}

    virtual void initialize(const StateSpace& state_space, const MyIndexType source_sindex_start, const MyIndexType source_sindex_end) = 0;

    virtual bool is_constant() const = 0;

    virtual double get_gamma_sum(const StateSpace& state_space, const MyIndexType source_sindex, const double t) const = 0;

    virtual CollapseGamma get_gamma(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex, const double t) const = 0;
    virtual bool is_gamma_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType target_sindex) const = 0;

    virtual std::unordered_set<std::string> get_outer_state_names(const StateSpace& state_space) const = 0;
    virtual CollapseGamma get_gamma_to_outer_state(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index, const double t) const = 0;
    virtual bool is_gamma_to_outer_state_nonzero(const StateSpace& state_space, const MyIndexType source_sindex, const MyIndexType outer_target_index) const = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) { }
};


class IStateSpaceIndexGetter
{
public:
    virtual ~IStateSpaceIndexGetter() { }

    virtual MyIndexType get_single_space_index(const MyIndexType sindex, const std::string& space_name) const = 0;
};


class StateSpaceRuntimeIndexGetter : public IStateSpaceIndexGetter
{
public:
    StateSpaceRuntimeIndexGetter(const std::vector<std::unique_ptr<const ISingleStateSpace>>& state_spaces);

    MyIndexType get_single_space_index(const MyIndexType sindex, const std::string& space_name) const final;

private:
    std::unordered_map<std::string, MyIndexType> map_spacename_to_n_dim_;
    std::unordered_map<std::string, MyIndexType> map_spacename_to_divider_;
};


class StateSpaceCachedIndexGetter : public IStateSpaceIndexGetter
{
public:
    StateSpaceCachedIndexGetter(const std::vector<std::unique_ptr<const ISingleStateSpace>>& state_spaces);

    MyIndexType get_single_space_index(const MyIndexType sindex, const std::string& space_name) const final;

private:
    std::unordered_map<std::string, std::vector<MyIndexType>> map_spacename_to_sindex_eigenvalueindex_;
};


class StateSpace
{
public:
    enum class IndexGetterType
    {
	Runtime,
	Cached,
    };

    StateSpace(
	    std::vector<std::unique_ptr<const ISingleStateSpace>>&& state_spaces,
	    std::vector<std::unique_ptr<const ICOperator>>&& c_operators,
	    const std::shared_ptr<IOperator>& hamiltonian,
	    IndexGetterType index_getter_type=IndexGetterType::Runtime
	);

    template<class EigenValueType>
    const SingleStateSpace<EigenValueType>& get_single_state_space(const std::string& space_name) const { return static_cast<const SingleStateSpace<EigenValueType>&>(map_name_to_space_.at(space_name).get()); }

    template<class EigenValueType>
    const EigenValueType& get_eigen_value(const MyIndexType sindex, const std::string& space_name) const;

    template<class EigenValueType>
    const EigenValueType& get_eigen_value_from_space(const MyIndexType index, const std::string& space_name) const;

    const std::string& get_outer_state_name(const MyIndexType index) const;

    MyIndexType get_n_dim_rho() const { return n_dim_rho_; }
    MyIndexType get_n_dim_phi() const { return n_dim_phi_; }

    std::vector<std::reference_wrapper<const ICOperator>> get_c_operators() const;
    const IOperator& get_hamiltonian() const { return *hamiltonian_; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    friend void save_construct_data(Archive& ar, const StateSpace* t, const unsigned int version)
    {
	ar << t->state_spaces_;
	ar << t->c_operators_;
	ar << t->hamiltonian_;
	ar << t->index_getter_type_;
    }

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);

    void initialize();

    std::vector<std::unique_ptr<const ISingleStateSpace>> state_spaces_;
    std::vector<std::unique_ptr<const ICOperator>> c_operators_;

    size_t n_space_;
    std::unordered_map<std::string, std::reference_wrapper<const ISingleStateSpace>> map_name_to_space_;
    std::unique_ptr<const IStateSpaceIndexGetter> index_getter_;

    size_t n_dim_rho_;

    std::vector<std::string> outer_state_names_;
    size_t n_dim_phi_;

    std::shared_ptr<IOperator> hamiltonian_;

    IndexGetterType index_getter_type_;
};


template<class EigenValueType>
const EigenValueType& StateSpace::get_eigen_value(const MyIndexType sindex, const std::string& space_name) const
{
    return get_eigen_value_from_space<EigenValueType>(index_getter_->get_single_space_index(sindex, space_name), space_name);
}


template<class EigenValueType>
const EigenValueType& StateSpace::get_eigen_value_from_space(const MyIndexType eigen_value_index, const std::string& space_name) const
{
    return get_single_state_space<EigenValueType>(space_name).get_eigen_value(eigen_value_index);
}


template<class Archive>
void StateSpace::serialize(Archive& ar, const unsigned int version)
{
    ar & state_spaces_;
    ar & c_operators_;
    ar & hamiltonian_;
    ar & index_getter_type_;
}


} // namespace cppoqss


namespace boost { namespace serialization {


template<class Archive, class EigenValueType>
inline void load_construct_data(
	Archive& ar, cppoqss::SingleStateSpace<EigenValueType>* t, const unsigned int version
    )
{
    std::string space_name_;
    std::vector<std::unique_ptr<EigenValueType>> eigen_values_;
    cppoqss::MyIndexType n_dim_;

    ar >> space_name_;
    ar >> eigen_values_;

    ::new(t)cppoqss::SingleStateSpace<EigenValueType>(space_name_, std::move(eigen_values_));
}


template<class Archive>
inline void load_construct_data(
	Archive& ar, cppoqss::StateSpace* t, const unsigned int version
    )
{
    std::vector<std::unique_ptr<const cppoqss::ISingleStateSpace>> state_spaces;
    std::vector<std::unique_ptr<const cppoqss::ICOperator>> c_operators;
    std::shared_ptr<cppoqss::IOperator> hamiltonian;
    cppoqss::StateSpace::IndexGetterType index_getter_type;

    ar >> state_spaces;
    ar >> c_operators;
    ar >> hamiltonian;
    ar >> index_getter_type;

    ::new(t)cppoqss::StateSpace(std::move(state_spaces), std::move(c_operators), hamiltonian, index_getter_type);
}


}} // namespace boost::serialization 


#endif
