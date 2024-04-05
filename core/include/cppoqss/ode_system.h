#ifndef ODE_SYSTEM_H
#define ODE_SYSTEM_H


#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/numeric/odeint/external/mpi/mpi.hpp>
#include <boost/operators.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/c_operator.h>
#include <cppoqss/density_matrix.h>
#include <cppoqss/operator.h>
#include <cppoqss/result.h>


namespace cppoqss {


class OdeLindbladMasterMPI
{
public:
    typedef boost::numeric::odeint::mpi_nested_algebra<boost::numeric::odeint::vector_space_algebra> algebra_type;

    class State
    : boost::multiplicative2<State, double
      >
    {
    public:
	typedef MyElementType value_type;
	const boost::mpi::communicator world;

	State();
	State(DensityMatrix& density_matrix);
	State(const State& rh);
	State(State&& rh);
	~State();

	void instantiate(const MyIndexType n_dim_rho, const MyIndexType n_dim_phi);

	State& operator()() { return *this; }
	const State& operator()() const { return *this; }

	State& operator=(const State& rh);
	State& operator=(State&& rh);

	State operator+(const State& rh); // Required in controlled_runge_kutta_spmat, which is a custom class
	State operator+(State&& rh);
	State& operator*=(const double rh);

	off_t dump_rho_append(const int save_mode, const std::filesystem::path& save_filepath) const;

	off_t dump_rho_rewrite(const std::filesystem::path& save_filepath) const;

	off_t dump_phi_append(const std::filesystem::path& save_filepath) const;

	off_t dump_phi_rewrite(const std::filesystem::path& save_filepath) const;

	MyMat& get_rho() { return *rho_; }
	const MyMat& get_rho() const { return *rho_; }
	MyMat* get_rho_ptr() { return rho_; }
	const MyMat* get_rho_ptr() const { return rho_; }
	MyVec& get_phi() { return *phi_; }
	const MyVec& get_phi() const { return *phi_; }
	MyVec* get_phi_ptr() { return phi_; }
	const MyVec* get_phi_ptr() const { return phi_; }

    private:
	MyMat* rho_;
	MyVec* phi_;

	bool is_own_resource_;
	bool is_pointing_density_matrix_;
    };

    class ErrorState
    : boost::additive2<ErrorState, double,
      boost::multiplicative2<ErrorState, double
      >>
    {
    friend ErrorState abs(const State& x);
    friend State operator/(ErrorState&& x, ErrorState&& y);

    public:
	ErrorState operator+(ErrorState &&rh);
	ErrorState& operator*=(const double rh);
	ErrorState& operator+=(const double rh);

    private:
	State state_;
	double added_factor_ = 0;
    };

    static std::vector<
	std::tuple<
	    std::string,
	    std::function<off_t(const State&, const int, const std::filesystem::path&)>,
	    std::function<off_t(const State&, const int, const std::filesystem::path&)>
	    >> get_results_saver();

    static const std::string type;
  
    OdeLindbladMasterMPI(DensityMatrix& density_matrix);

    void operator()(const State& x, State& dxdt, const double t);
    void pre_observation_process(State& x, const double t);
    void post_observation_process(State& x, const double t);

    DensityMatrix& get_state() { return density_matrix_; }
    const DensityMatrix& get_state() const { return density_matrix_; }
    DensityMatrix& get_density_matrix() { return density_matrix_; }
    const DensityMatrix& get_density_matrix() const { return density_matrix_; }
    double get_t0_for_multiply_collapse_prob() const { return t0_for_multiply_collapse_prob_; }

    void multiply_constant_collapse_prob(State& state, const double t) const;

private:
    struct AnalyzedCOperator
    {
	AnalyzedCOperator(const std::reference_wrapper<const ICOperator>& c_operator) : c_operator(c_operator) { }

	const std::reference_wrapper<const ICOperator> c_operator;
	std::unordered_set<MyIndexType> state_with_nonzero_constant_collapse;
	std::unordered_set<MyIndexType> state_with_nonzero_variable_collapse;
	std::unordered_map<MyIndexType, std::unordered_set<MyIndexType>> nonzero_collapse_pairs;
	std::unordered_map<MyIndexType, std::unordered_set<MyIndexType>> nonzero_collapse_pairs_to_outer_state;
    };

    void evaluate_hamiltonian(const double t) const;
    void multiply_constant_collapse_prob(MyMat& mat, const double t, bool is_inverting_prob) const;
    void fill_orig_rho(const State& x, const double t) const;
    void fill_liouvillian(const double t, MyMat& Drho_Dt, MyVec& Dphi_Dt, const std::vector<const MyMat*>& nonzero_place_source_for_Drho_Dt) const;

    DensityMatrix& density_matrix_;
    const IOperator& hamiltonian_;

    mutable MyMat Hmat_;
    SparseMatrixNonzeroStruct Hmat_nonzero_struct_;

    std::vector<AnalyzedCOperator> analyzed_c_operators_;

    mutable MyMat orig_rho_;
    mutable MyVec diag_rho_;

    mutable double t0_for_multiply_collapse_prob_;

    MyIndexType rho_n_dim_;
    MyIndexType rho_n_ownership_row_;
    std::array<MyIndexType, 2> rho_ownership_range_row_;
    std::vector<std::array<MyIndexType, 2>> rho_ownership_range_row_of_each_rank_;

    MyIndexType phi_n_dim_;
    MyIndexType phi_n_ownership_;
    std::array<MyIndexType, 2> phi_ownership_range_;
    std::vector<std::array<MyIndexType, 2>> phi_ownership_range_of_each_rank_;
};


OdeLindbladMasterMPI::ErrorState abs(const OdeLindbladMasterMPI::State& x);


OdeLindbladMasterMPI::State operator/(OdeLindbladMasterMPI::ErrorState&& x, OdeLindbladMasterMPI::ErrorState&& y);


} // namespace cppoqss


template<>
struct boost::numeric::odeint::vector_space_norm_inf<cppoqss::OdeLindbladMasterMPI::State>
{
    typedef double result_type;

    result_type operator()(const cppoqss::OdeLindbladMasterMPI::State& x) const
    {
	double max_val = 0.;

	const cppoqss::MyMat& rho = x.get_rho();
	rho.loop_over_nonzero_elements(
		[&max_val](const cppoqss::MyIndexType i, const cppoqss::MyIndexType j, const cppoqss::MyElementType element) {
		    max_val = std::max(max_val, std::abs(element));
		}
	    );

	const cppoqss::MyVec& phi = x.get_phi();
	phi.loop_over_elements(
		[&max_val](const cppoqss::MyIndexType i, const cppoqss::MyElementType element) {
		    max_val = std::max(max_val, std::abs(element));
		}
	    );

	return max_val;
    }
};


#endif
