#ifndef DENSITY_MATRIX_H
#define DENSITY_MATRIX_H


#include <functional>
#include <memory>
#include <vector>

#include <boost/mpi.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_space.h>


namespace cppoqss {


class DensityMatrix
{
public:
    typedef std::function<MyElementType(const StateSpace&, const MyIndexType, const MyIndexType)> InitializerType;

    static const std::string type;

    DensityMatrix(
	    const std::shared_ptr<const StateSpace>& state_space,
	    InitializerType rho_element_getter
	    );

    MyIndexType get_n_dim_rho() const { return n_dim_rho_; }
    MyIndexType get_n_dim_phi() const { return n_dim_phi_; }
    MyMat& get_rho() { return rho_; }
    const MyMat& get_rho() const { return rho_; }
    MyVec& get_phi() { return phi_; }
    const MyVec& get_phi() const { return phi_; }
    double get_t() const { return t_; }

private:
    const MyIndexType n_dim_rho_;
    const MyIndexType n_dim_phi_;
    MyMat rho_;
    MyVec phi_;
    double t_;

public:
    const StateSpace& state_space_;
    const MyMat& const_rho_;
    const MyVec& const_phi_;
};


} // namespace cppoqss


#endif
