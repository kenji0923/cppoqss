#include <cppoqss/operator.h>

#include <cppoqss/arithmetic.h>


namespace cppoqss {


template<class Archive>
void special_operator::Zero::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperator>(*this);
}


template void special_operator::Zero::serialize(boost::archive::text_iarchive& ar, const unsigned int version);
template void special_operator::Zero::serialize(boost::archive::text_oarchive& ar, const unsigned int version);


template<class Archive>
void special_operator::One::serialize(Archive& ar, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperator>(*this);
}


template void special_operator::One::serialize(boost::archive::text_iarchive& ar, const unsigned int version);
template void special_operator::One::serialize(boost::archive::text_oarchive& ar, const unsigned int version);


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


template<class Archive>
void save_construct_data(Archive& ar, const SumOperator* t, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperator>(*t);
    ar << t->op1_;
    ar << t->op2_;
}


template<class Archive>
void load_construct_data(Archive& ar, cppoqss::SumOperator* t, const unsigned int version)
{
    std::shared_ptr<const cppoqss::IOperator> op1_;
    std::shared_ptr<const cppoqss::IOperator> op2_;

    ar >> op1_;
    ar >> op2_;

    ::new(t)cppoqss::SumOperator(op1_, op2_);
}


template void save_construct_data(boost::archive::text_oarchive& ar, const SumOperator* t, const unsigned int version);
template void load_construct_data(boost::archive::text_iarchive& ar, SumOperator* t, const unsigned int version);


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


template<class Archive>
void save_construct_data(Archive& ar, const ProductOperator* t, const unsigned int version)
{
    ar & boost::serialization::base_object<IOperator>(*t);
    ar << t->op1_;
    ar << t->op2_;
}


template<class Archive>
void load_construct_data(Archive& ar, cppoqss::ProductOperator* t, const unsigned int version)
{
    std::shared_ptr<const cppoqss::IOperator> op1_;
    std::shared_ptr<const cppoqss::IOperator> op2_;

    ar >> op1_;
    ar >> op2_;

    ::new(t)cppoqss::ProductOperator(op1_, op2_);
}


template void save_construct_data(boost::archive::text_oarchive& ar, const ProductOperator* t, const unsigned int version);
template void load_construct_data(boost::archive::text_iarchive& ar, ProductOperator* t, const unsigned int version);


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
