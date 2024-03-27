#include <memory>
#include <string>
#include <vector>

#include <boost/serialization/base_object.hpp>


class ISingleStateSpace;


namespace cppoqss {


template<class EigenValueType> class SingleStateSpace;


template<class Archive, class EigenValueType> void save_construct_data(Archive& ar, const SingleStateSpace<EigenValueType>* t, const unsigned int version);


template<class Archive, class EigenValueType> void load_construct_data(Archive& ar, SingleStateSpace<EigenValueType>* t, const unsigned int version);


template<class Archive, class EigenValueType>
void save_construct_data(Archive& ar, const SingleStateSpace<EigenValueType>* t, const unsigned int version)
{
    ar & boost::serialization::base_object<ISingleStateSpace>(*t);
    ar << t->space_name_;
    ar << t->eigen_values_;
}


template<class Archive, class EigenValueType>
void load_construct_data(Archive& ar, SingleStateSpace<EigenValueType>* t, const unsigned int version)
{
    std::string space_name_;
    std::vector<std::unique_ptr<EigenValueType>> eigen_values_;

    ar >> space_name_;
    ar >> eigen_values_;

    ::new(t)cppoqss::SingleStateSpace<EigenValueType>(space_name_, std::move(eigen_values_));
}


}
