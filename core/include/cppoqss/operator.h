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

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>

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
    mutable MyElementType evaluated_shared_term_;
};


template<class StatePairInfoType>
void IOperatorImplWithSharedTerm<StatePairInfoType>::set_time(const double t) const
{
    evaluated_shared_term_ = this->get_shared_term(t);
}


template<class StatePairInfoType>
class RuntimeCalculatingOperatorWithSharedTerm : public IOperatorImplWithSharedTerm<StatePairInfoType>
{
public:
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;
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


namespace special_operator
{


class Zero: public IOperator
{
public:
    void set_time(const double t) const override { }
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override { return 0.0; }
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override { return false; }

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive& ar, const std::uint32_t version)
    { }
};


class One : public IOperator
{
public:
    void set_time(const double t) const override { }
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override { return 1.0; }
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override { return true; }

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive& ar, const std::uint32_t version)
    { }
};


} // namespace special_operator


class SumOperator : public IOperator 
{
public:
    SumOperator(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2) : op1_(op1), op2_(op2) { }

    void set_time(const double t) const override;
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive& ar, const std::uint32_t version)
    {
	ar(op1_, op2_);
    }

    template<class Archive>
    static void load_and_construct(Archive& ar, cereal::construct<SumOperator>& construct, const std::uint32_t version)
    {
	std::shared_ptr<const IOperator> op1_;
	std::shared_ptr<const IOperator> op2_;

	ar(op1_, op2_);

	construct(op1_, op2_);
    }

    std::shared_ptr<const IOperator> op1_;
    std::shared_ptr<const IOperator> op2_;
};


class ProductOperator : public IOperator 
{
public:
    ProductOperator(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2) : op1_(op1), op2_(op2) { }

    void set_time(const double t) const override;
    MyElementType evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const override;
    bool is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const override;

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive& ar, const std::uint32_t version)
    {
	ar(op1_, op2_);
    }

    template<class Archive>
    static void load_and_construct(Archive& ar, cereal::construct<ProductOperator>& construct, const std::uint32_t version)
    {
	std::shared_ptr<const IOperator> op1_;
	std::shared_ptr<const IOperator> op2_;

	ar(op1_, op2_);

	construct(op1_, op2_);
    }

    std::shared_ptr<const IOperator> op1_;
    std::shared_ptr<const IOperator> op2_;
};


std::shared_ptr<IOperator> operator+(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2);


std::shared_ptr<IOperator> operator*(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2);


} // namespace cppoqss


#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>


CEREAL_CLASS_VERSION(cppoqss::special_operator::Zero, 1)

CEREAL_CLASS_VERSION(cppoqss::special_operator::One, 1)

CEREAL_CLASS_VERSION(cppoqss::SumOperator, 1)

CEREAL_CLASS_VERSION(cppoqss::ProductOperator, 1)


CEREAL_REGISTER_TYPE(cppoqss::special_operator::Zero)

CEREAL_REGISTER_TYPE(cppoqss::special_operator::One)

CEREAL_REGISTER_TYPE(cppoqss::SumOperator)

CEREAL_REGISTER_TYPE(cppoqss::ProductOperator)


CEREAL_REGISTER_POLYMORPHIC_RELATION(cppoqss::IOperator, cppoqss::special_operator::Zero)

CEREAL_REGISTER_POLYMORPHIC_RELATION(cppoqss::IOperator, cppoqss::special_operator::One)

CEREAL_REGISTER_POLYMORPHIC_RELATION(cppoqss::IOperator, cppoqss::SumOperator)

CEREAL_REGISTER_POLYMORPHIC_RELATION(cppoqss::IOperator, cppoqss::ProductOperator)


#endif
