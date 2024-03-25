#include <cppoqss/operator.h>

#include <cppoqss/arithmetic.h>


namespace cppoqss {


void SumOperator::set_time(const double t) const
{
    op1_->set_time(t);
    op2_->set_time(t);
}


MyElementType SumOperator::evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const
{
    return op1_->evaluate_element(state_space, bra_sindex, ket_sindex, t) + op2_->evaluate_element(state_space, bra_sindex, ket_sindex, t);
}


bool SumOperator::is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const
{
    return op1_->is_element_nonzero(state_space, bra_sindex, ket_sindex) || op2_->is_element_nonzero(state_space, bra_sindex, ket_sindex);
}


void ProductOperator::set_time(const double t) const
{
    op1_->set_time(t);
    op2_->set_time(t);
}


MyElementType ProductOperator::evaluate_element(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex, const double t) const
{
    return op1_->evaluate_element(state_space, bra_sindex, ket_sindex, t) * op2_->evaluate_element(state_space, bra_sindex, ket_sindex, t);
}


bool ProductOperator::is_element_nonzero(const StateSpace& state_space, const MyIndexType bra_sindex, const MyIndexType ket_sindex) const
{
    return op1_->is_element_nonzero(state_space, bra_sindex, ket_sindex) && op2_->is_element_nonzero(state_space, bra_sindex, ket_sindex);
}


std::shared_ptr<IOperator> operator+(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2)
{
    return std::make_shared<SumOperator>(op1, op2);
}


std::shared_ptr<IOperator> operator*(const std::shared_ptr<const IOperator>& op1, const std::shared_ptr<const IOperator>& op2)
{
    return std::make_shared<ProductOperator>(op1, op2);
}


} // namespace cppoqss


#include <boost/serialization/export.hpp>


BOOST_CLASS_EXPORT_IMPLEMENT(cppoqss::special_operator::Zero)
BOOST_CLASS_EXPORT_IMPLEMENT(cppoqss::special_operator::One)
BOOST_CLASS_EXPORT_IMPLEMENT(cppoqss::SumOperator)
BOOST_CLASS_EXPORT_IMPLEMENT(cppoqss::ProductOperator)
