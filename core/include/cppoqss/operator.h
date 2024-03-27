#ifndef OPERATOR_H
#define OPERATOR_H


#include <array>
#include <cfloat>
#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_space.h>


namespace cppoqss {


/**
 * Virtual methods must be implemented by user.
 */
template<class StatePairInfoType>
class IOperatorImplWithSharedTerm : public IOperator
{
public:
    virtual ~IOperatorImplWithSharedTerm() { }

    void set_time(const double t) const override;

    MyElementType get_evaluated_shared_term() const { return evaluated_shared_term_; }

    virtual StatePairInfoType get_state_pair_info(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const = 0;
    virtual MyElementType get_each_element(const StatePairInfoType& state_pair_info, const double t) const = 0;
    virtual MyElementType get_shared_term(const double t) const = 0;
    virtual bool is_each_element_nonzero(const StatePairInfoType& state_pair_info) const = 0;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);

    mutable MyElementType evaluated_shared_term_;
};


template<class StatePairInfoType>
void IOperatorImplWithSharedTerm<StatePairInfoType>::set_time(const double t) const
{
    evaluated_shared_term_ = this->get_shared_term(t);
}

template<class StatePairInfoType>
template<class Archive>
void IOperatorImplWithSharedTerm<StatePairInfoType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperator>(*this);
}


template<class StatePairInfoType>
class RuntimeCalculatingOperatorWithSharedTerm : public IOperatorImplWithSharedTerm<StatePairInfoType>
{
public:
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);
};


template<class StatePairInfoType>
MyElementType RuntimeCalculatingOperatorWithSharedTerm<StatePairInfoType>::evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const
{
    StatePairInfoType state_pair_info = this->get_state_pair_info(state_space, bra_sindex, ket_sindex);
    return this->get_evaluated_shared_term() * this->get_each_element(state_pair_info, t);
}


template<class StatePairInfoType>
bool RuntimeCalculatingOperatorWithSharedTerm<StatePairInfoType>::is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const
{
    StatePairInfoType state_pair_info = this->get_state_pair_info(state_space, bra_sindex, ket_sindex);
    return this->is_each_element_nonzero(state_pair_info);
}


template<class StatePairInfoType>
template<class Archive>
void RuntimeCalculatingOperatorWithSharedTerm<StatePairInfoType>::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperatorImplWithSharedTerm<StatePairInfoType>>(*this);
}


namespace special_operator
{


class Zero: public IOperator
{
public:
    void set_time(const double t) const override { }
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override { return 0.0; }
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override { return false; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);
};


extern template void Zero::serialize(boost::archive::text_iarchive& ar, const unsigned int version);
extern template void Zero::serialize(boost::archive::text_oarchive& ar, const unsigned int version);


class One : public IOperator
{
public:
    void set_time(const double t) const override { }
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override { return 1.0; }
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override { return true; }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version);
};


extern template void One::serialize(boost::archive::text_iarchive& ar, const unsigned int version);
extern template void One::serialize(boost::archive::text_oarchive& ar, const unsigned int version);


} // namespace special_operator


class SumOperator;


template<class Archive> void save_construct_data(Archive& ar, const SumOperator* t, const unsigned int version);
template<class Archive> void load_construct_data(Archive& ar, SumOperator* t, const unsigned int version);


class SumOperator : public IOperator 
{
public:
    SumOperator(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2) : op1_(op1), op2_(op2) { }

    void set_time(const double t) const override;
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;

private:
    friend class boost::serialization::access;

    template<class Archive>
    friend void save_construct_data(Archive& ar, const SumOperator* t, const unsigned int version);

    template<class Archive>
    friend void load_construct_data(Archive& ar, cppoqss::SumOperator* t, const unsigned int version);

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) { }

    std::shared_ptr<const IOperator> op1_;
    std::shared_ptr<const IOperator> op2_;
};


extern template void save_construct_data(boost::archive::text_oarchive& ar, const SumOperator* t, const unsigned int version);
extern template void load_construct_data(boost::archive::text_iarchive& ar, SumOperator* t, const unsigned int version);


class ProductOperator;


template<class Archive> void save_construct_data(Archive& ar, const ProductOperator* t, const unsigned int version);
template<class Archive> void load_construct_data(Archive& ar, ProductOperator* t, const unsigned int version);


class ProductOperator : public IOperator 
{
public:
    ProductOperator(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2) : op1_(op1), op2_(op2) { }

    void set_time(const double t) const override;
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;

private:
    friend class boost::serialization::access;

    template<class Archive>
    friend void save_construct_data(Archive& ar, const ProductOperator* t, const unsigned int version);

    template<class Archive>
    friend void load_construct_data(Archive& ar, cppoqss::ProductOperator* t, const unsigned int version);

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) { }

    std::shared_ptr<const IOperator> op1_;
    std::shared_ptr<const IOperator> op2_;
};


extern template void save_construct_data(boost::archive::text_oarchive& ar, const ProductOperator* t, const unsigned int version);
extern template void load_construct_data(boost::archive::text_iarchive& ar, ProductOperator* t, const unsigned int version);


std::shared_ptr<IOperator> operator+(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2);


std::shared_ptr<IOperator> operator*(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2);


} // namespace cppoqss


#include <boost/serialization/export.hpp>


BOOST_CLASS_EXPORT_KEY(cppoqss::special_operator::Zero)
BOOST_CLASS_EXPORT_KEY(cppoqss::special_operator::One)
BOOST_CLASS_EXPORT_KEY(cppoqss::SumOperator)
BOOST_CLASS_EXPORT_KEY(cppoqss::ProductOperator)


#endif
