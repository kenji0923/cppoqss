#include <cppoqss/ode_system.h>

#include <complex>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/c_operator.h>
#include <cppoqss/density_matrix.h>
#include <cppoqss/logger.h>
#include <cppoqss/mpi_helper.h>
#include <cppoqss/operator.h>
#include <cppoqss/progress_bar.h>
#include <cppoqss/state_system_name.h>
#include <cppoqss/unit.h>


#define INCLUDE_CONSTANTCOLLAPSE_TO_RHO


namespace cppoqss {


using State = OdeLindbladMasterMPI::State;
using ErrorState = OdeLindbladMasterMPI::ErrorState;


State::State()
: rho_(nullptr), phi_(nullptr), is_own_resource_(false), is_pointing_density_matrix_(false)
{ }


State::State(DensityMatrix& density_matrix)
: rho_(&density_matrix.get_rho()), phi_(&density_matrix.get_phi()), is_own_resource_(false), is_pointing_density_matrix_(true)
{ }


State::State(const State& rh)
: rho_(nullptr), phi_(nullptr), is_own_resource_(false), is_pointing_density_matrix_(false)
{
    *this = rh;
}


State::State(State&& rh)
: rho_(nullptr), phi_(nullptr), is_own_resource_(false), is_pointing_density_matrix_(false)
{
    *this = std::move(rh);
}


State::~State()
{
  if (is_own_resource_) {
    delete rho_;
    delete phi_;
  }
}


void State::instantiate(const MyIndexType n_dim_rho, const MyIndexType n_dim_phi)
{
    if (is_own_resource_) {
	delete rho_;
	delete phi_;
    }

    rho_ = new MyMat(n_dim_rho, true);
    phi_ = new MyVec(n_dim_phi);

    is_own_resource_ = true;
    is_pointing_density_matrix_ = false;
}


State& State::operator=(const State& rh)
{
    if (is_own_resource_) {
	delete rho_;
	delete phi_;
    }

    if (rh.is_pointing_density_matrix_ || rh.is_own_resource_) {
	rho_ = new MyMat(rh.rho_->duplicate());
	phi_ = new MyVec(rh.phi_->duplicate());
	is_own_resource_ = true;
	is_pointing_density_matrix_ = false;
    } else {
	rho_ = rh.rho_;
	phi_= rh.phi_;
	is_own_resource_ = false;
	is_pointing_density_matrix_ = false;
    }

    return *this;
}


State& State::operator=(State&& rh)
{
  if (is_own_resource_) {
    delete rho_;
    delete phi_;
  }

  rho_ = rh.rho_;
  phi_ = rh.phi_;
  is_own_resource_ = rh.is_own_resource_;
  is_pointing_density_matrix_ = rh.is_pointing_density_matrix_;

  rh.rho_ = nullptr;
  rh.phi_ = nullptr;
  rh.is_own_resource_ = false;
  rh.is_pointing_density_matrix_ = false;

  return *this;
}


State State::operator+(const State& rh)
{
  State result(rh);
  result.rho_->add_ax(1.0, *this->rho_);
  result.phi_->add_ax(1.0, *this->phi_);

  return result;
}


State State::operator+(State&& rh)
{
  State result(std::move(rh));
  result.rho_->add_ax(1.0, *this->rho_);
  result.phi_->add_ax(1.0, *this->phi_);

  return result;
}


State& State::operator*=(const double rh)
{
    rho_->scale(rh);
    phi_->scale(rh);

    return *this;
}


off_t State::dump_rho_append(const int save_mode, const std::filesystem::path& save_filepath) const
{
    if (save_mode == 0) {
	return rho_->append_to_file(save_filepath);
    } else if (save_mode == 1) {
	return rho_->get_diagonal().append_to_file(save_filepath);
    } else {
	fprintf(stderr, "undefined save mode was requested\n");
	exit(1);
    }
}


off_t State::dump_rho_rewrite(const std::filesystem::path& save_filepath) const
{
    rho_->write_to_file(save_filepath);
    return 0;
}


off_t State::dump_phi_append(const std::filesystem::path& save_filepath) const
{
    return phi_->append_to_file(save_filepath);
}


off_t State::dump_phi_rewrite(const std::filesystem::path& save_filepath) const
{
    phi_->write_to_file(save_filepath);
    return 0;
}


ErrorState ErrorState::operator+(ErrorState&& rh)
{
  ErrorState result(std::move(*this));
  result.state_ = result.state_ +  std::move(rh.state_);
  result.added_factor_ += rh.added_factor_;

  return result;
}


ErrorState& ErrorState::operator*=(const double rh)
{
  state_ *= rh;
  added_factor_ *= rh;
  return *this;
}


ErrorState& ErrorState::operator+=(const double rh)
{
  added_factor_ += rh;
  return *this;
}


std::vector<
    std::tuple<
	std::string,
	std::function<off_t(const State&, const int, const std::filesystem::path&)>,
	std::function<off_t(const State&, const int, const std::filesystem::path&)>
	>> OdeLindbladMasterMPI::get_results_saver()
{
    return {
	    {
		"rho",
		[](const State& state, const int save_mode, const std::filesystem::path& save_filepath){ return state.dump_rho_append(save_mode, save_filepath); },
		[](const State& state, const int save_mode, const std::filesystem::path& save_filepath){ return state.dump_rho_rewrite(save_filepath); }
	    },
	    {
		"phi",
		[](const State& state, const int save_mode, const std::filesystem::path& save_filepath){ return state.dump_phi_append(save_filepath); },
		[](const State& state, const int save_mode, const std::filesystem::path& save_filepath){ return state.dump_phi_rewrite(save_filepath); }
	    },
	};
}


const std::string OdeLindbladMasterMPI::type = CPPOQSS_MACRO_SYSTEM_NAME_ODELINDBLAD;


OdeLindbladMasterMPI::OdeLindbladMasterMPI(DensityMatrix& density_matrix)
: density_matrix_(density_matrix), hamiltonian_(density_matrix_.state_space_.get_hamiltonian()), Hmat_(density_matrix_.get_n_dim_rho(), true), orig_rho_(nullptr), diag_rho_(density_matrix_.get_n_dim_rho()), t0_for_multiply_collapse_prob_(density_matrix.get_t())
{ 
    if (mpi_helper::is_printing_rank()) {
	printf("initialize ODE system\n");
    }

    // 
    // Prepare Hamiltonian matrix
    // 
    rho_n_dim_ = density_matrix_.get_n_dim_rho();
    rho_ownership_range_row_ = Hmat_.get_ownership_range_row();
    rho_n_ownership_row_ = rho_ownership_range_row_[1] - rho_ownership_range_row_[0];

    phi_n_dim_ = density_matrix_.get_n_dim_phi();

    MyIndexType current_row = -1;
    ProgressBar::InitBar("prepare Hamiltonian", rho_n_ownership_row_);

    auto Hmat_nonzero_checker = [this, &current_row](const MyIndexType i, const MyIndexType j)
    {
	const MyIndexType ii = i - Hmat_.get_ownership_range_row()[0];
	if (Hmat_nonzero_struct_.get_n_row() <= ii) {
	    Hmat_nonzero_struct_.push_back_row();
	}

	bool is_element_nonzero = hamiltonian_.is_element_nonzero(density_matrix_.state_space_, i, j);
	if (is_element_nonzero) {
	    Hmat_nonzero_struct_.push_back_element(j);
	}

	if (current_row != i) {
	    if (current_row >= 0) ProgressBar::ProgressStep();
	    current_row = i;
	}

	return is_element_nonzero;
    };

    const auto Hmat_nonzero_numbers = Hmat_.get_nonzero_numbers(Hmat_nonzero_checker);
    Hmat_nonzero_struct_.push_back_row();
    ProgressBar::ProgressStep();

    Hmat_.set_preallocation(0, &Hmat_nonzero_numbers.diag[0], 0, &Hmat_nonzero_numbers.nondiag[0]);

    evaluate_hamiltonian(0);

    const MyIndexType Hmat_number_of_nonzeros = Hmat_.get_number_of_nonzeros();

    if (mpi_helper::is_printing_rank()) {
	printf("Hamiltonian non-zeros: %ld\n", Hmat_number_of_nonzeros);
    }


    // 
    // Scan collapse
    // 
    for (const auto& c_operator_ref : density_matrix_.state_space_.get_c_operators()) {
	auto& c_operator = const_cast<ICOperator&>(c_operator_ref.get());
	c_operator.initialize(density_matrix_.state_space_, rho_ownership_range_row_[0], rho_ownership_range_row_[1]);

	analyzed_c_operators_.emplace_back(c_operator_ref);
    }


    ProgressBar::InitBar("scan c-operator", rho_n_ownership_row_);

    for (MyIndexType i = rho_ownership_range_row_[0]; i < rho_ownership_range_row_[1]; ++i) {
	for (auto& analyzed_c_operator : analyzed_c_operators_) {
	    const ICOperator& c_op = analyzed_c_operator.c_operator.get();

	    for (MyIndexType j = 0; j < rho_n_dim_; ++j) {
		const bool is_gamma_nonzero = c_op.is_gamma_nonzero(density_matrix_.state_space_, i, j);

		if (is_gamma_nonzero) {
		    if (c_op.is_constant()) analyzed_c_operator.state_with_nonzero_constant_collapse.emplace(i);
		    else analyzed_c_operator.state_with_nonzero_variable_collapse.emplace(i);

		    analyzed_c_operator.nonzero_collapse_pairs.try_emplace(i);
		    analyzed_c_operator.nonzero_collapse_pairs.at(i).emplace(j);

		    // analyzed_c_operator.cached_gamma.try_emplace(i);
		    // analyzed_c_operator.cached_gamma.at(i).emplace(j, c_op.get_gamma(density_matrix_.state_space_, i, j, 0).get_fill_gamma());
		}
	    }

	    for (MyIndexType j = 0; j < phi_n_dim_; ++j) {
		const bool is_gamma_nonzero = c_op.is_gamma_to_outer_state_nonzero(density_matrix_.state_space_, i, j);

		if (is_gamma_nonzero) {
		    if (c_op.is_constant()) analyzed_c_operator.state_with_nonzero_constant_collapse.emplace(i);
		    else analyzed_c_operator.state_with_nonzero_variable_collapse.emplace(i);

		    analyzed_c_operator.nonzero_collapse_pairs_to_outer_state.try_emplace(i);
		    analyzed_c_operator.nonzero_collapse_pairs_to_outer_state.at(i).emplace(j);
		}
	    }
	}

	ProgressBar::ProgressStep();
    }

    size_t n_collapse_pair = 0;

    auto count_collapse_pair = [&n_collapse_pair](const std::unordered_map<MyIndexType, std::unordered_set<MyIndexType>>& collapse_pair)
    {
	for (const auto& pair_i_jset : collapse_pair) {
	    n_collapse_pair += pair_i_jset.second.size();
	}
    };

    for (const auto& analyzed_c_operator : analyzed_c_operators_) {
	count_collapse_pair(analyzed_c_operator.nonzero_collapse_pairs);
	count_collapse_pair(analyzed_c_operator.nonzero_collapse_pairs_to_outer_state);
    }

    n_collapse_pair = boost::mpi::all_reduce(mpi_helper::world, n_collapse_pair, std::plus<size_t>());

    if (mpi_helper::is_printing_rank()) {
	printf("collapse pairs: %zu\n", n_collapse_pair);
    }

    //
    // Register log stages
    //
    Logger::kLogger.add_stage("DxDtOther");
    Logger::kLogger.add_stage("EvalH");
    Logger::kLogger.add_stage("HByRho");
    Logger::kLogger.add_stage("FillL");
}


void OdeLindbladMasterMPI::operator()(const State& x, State& dxdt, const double t)
{
    Logger::kLogger.push("DxDtOther");

    dxdt.instantiate(x.get_rho().get_n_dim(), x.get_phi().get_n_dim());
    MyMat& Drho_Dt = dxdt.get_rho();
    MyVec& Dphi_Dt = dxdt.get_phi();

    Logger::kLogger.push("EvalH");
    evaluate_hamiltonian(t); 
    Logger::kLogger.pop();

    fill_orig_rho(x, t);

    Logger::kLogger.push("HByRho");
    MyMat HByRho = MyMat::MatMatMulti(Hmat_, orig_rho_);
    MyMat RhoByH = MyMat::HermitianTranspose(HByRho);
    Logger::kLogger.pop();


    Logger::kLogger.push("FillL");
    fill_liouvillian(t, Drho_Dt, Dphi_Dt, {&orig_rho_, &HByRho, &RhoByH});
    Logger::kLogger.pop();

    Drho_Dt.add_ax(1.0 / std::complex<double>(0, u::hbar), HByRho);
    Drho_Dt.add_ax(-1.0 / std::complex<double>(0, u::hbar), RhoByH);

#ifdef INCLUDE_CONSTANTCOLLAPSE_TO_RHO
    multiply_constant_collapse_prob(Drho_Dt, t, true);
#endif

    Logger::kLogger.pop();
}


void OdeLindbladMasterMPI::pre_observation_process(State& x, const double t)
{
#ifdef INCLUDE_CONSTANTCOLLAPSE_TO_RHO
  multiply_constant_collapse_prob(x, t);
#endif
}


void OdeLindbladMasterMPI::post_observation_process(State& x, const double t)
{
#ifdef INCLUDE_CONSTANTCOLLAPSE_TO_RHO
  t0_for_multiply_collapse_prob_= t;
#endif
}


void OdeLindbladMasterMPI::multiply_constant_collapse_prob(State& state, const double t) const
{
  multiply_constant_collapse_prob(state.get_rho(), t, false);
}


void OdeLindbladMasterMPI::evaluate_hamiltonian(const double t) const
{
    hamiltonian_.set_time(t);

    Hmat_nonzero_struct_.loop_by_row(
	    [this, &t](const MyIndexType ii, const MyIndexType n_element, const MyIndexType* cols) {
	    const MyIndexType i = Hmat_.get_ownership_range_row()[0] + ii;

	    std::vector<MyElementType> elements(n_element);
	    for (MyIndexType i_element = 0; i_element < n_element; ++i_element) {
	    const MyIndexType j = cols[i_element];
	    elements[i_element] = hamiltonian_.evaluate_element(density_matrix_.state_space_, i, j, t);
	    }

	    Hmat_.set_values(1, &i, n_element,  cols, &elements[0]);
	    }
	    );
    Hmat_.begin_final_assembly();
    Hmat_.end_final_assembly();
}


void OdeLindbladMasterMPI::multiply_constant_collapse_prob(MyMat& mat, const double t, bool is_inverting_prob) const
{
  const double t0 = t0_for_multiply_collapse_prob_;
  const double sign = is_inverting_prob ? 1.0 : -1.0;

  std::unordered_map<MyIndexType, double> constant_gamma;

  for (const auto& analyzed_c_operator : analyzed_c_operators_) {
      for (const auto& i : analyzed_c_operator.state_with_nonzero_constant_collapse) {
	  const auto& c_op = analyzed_c_operator.c_operator.get();
	  if (c_op.is_constant()) {
	      constant_gamma.try_emplace(i);
	      constant_gamma.at(i) += c_op.get_gamma_sum(density_matrix_.state_space_, i, t);
	  }
      }
  }

  std::unordered_map<MyIndexType, MyRowCopied> rows_copied;
  for (MyIndexType i = mat.get_ownership_range_row()[0]; i < mat.get_ownership_range_row()[1]; ++i) {
    MyRow row = mat.get_row(i);
    rows_copied.emplace(i, row.copy());
    mat.restore_row(i, row);
  }

  // TODO avoid reading matrix every time
  for (size_t rank = 0; rank < mpi_helper::get_size(); ++rank) {
    std::unordered_map<MyIndexType, double> constant_gamma_received;
    if (rank == mpi_helper::get_rank()) {
      constant_gamma_received = std::move(constant_gamma);
    }
    boost::mpi::broadcast(mpi_helper::world, constant_gamma_received, rank);

    for (auto& pair_i_row : rows_copied) {
      const MyIndexType i = pair_i_row.first;
      MyRowCopied& row = pair_i_row.second;

      double gamma_i = 0;
      if (constant_gamma_received.find(i) != constant_gamma_received.end()) {
        gamma_i = constant_gamma_received.at(i);
      }

      for (MyIndexType jj = 0; jj < row.n_col; ++jj) {
        const MyIndexType j = row.cols[jj];

        double gamma_j = 0;
        if (constant_gamma_received.find(j) != constant_gamma_received.end()) {
          gamma_j = constant_gamma_received.at(j);
        }

        const double gamma = gamma_i + gamma_j;

        row.values[jj] *= std::exp(sign * 0.5 * gamma * (t - t0));
      }
    }
  }

  for (const auto& pair_i_row : rows_copied) {
    const MyIndexType i = pair_i_row.first;
    const MyRowCopied& row = pair_i_row.second;

    mat.set_values(1, &i, row.n_col, &row.cols[0], &row.values[0]);
  }

  mat.begin_final_assembly();
  mat.end_final_assembly();
}


void OdeLindbladMasterMPI::fill_orig_rho(const State& x, const double t) const
{
#ifdef INCLUDE_CONSTANTCOLLAPSE_TO_RHO
  orig_rho_ = x.get_rho();
  multiply_constant_collapse_prob(orig_rho_, t, false);
#else
  // TODO implement
  // orig_rho_.point(x.get_rho());
#endif
}


void OdeLindbladMasterMPI::fill_liouvillian(const double t, MyMat& Drho_Dt, MyVec& Dphi_Dt, const std::vector<const MyMat*>& nonzero_place_source_for_Drho_Dt) const
{
    //  
    // Collapse rates.
    // i -> . with gamma.
    // Mapping i: gamma.
    //  
    std::unordered_map<MyIndexType, double> gamma_from_i;
  
    // 
    // Fill rates.
    // . -> j with rate.
    // Mapping j: rate.
    // 
    std::unordered_map<MyIndexType, double> rate_to_j;
    std::unordered_map<MyIndexType, double> rate_to_outer_j;

    // 
    // Calculate gamma and rates.
    // 
    MyVec rho_diag_part_vec = orig_rho_.get_diagonal();

    MyRow rho_diag_part_row = rho_diag_part_vec.get_row();

    for (const auto& analyzed_c_operator : analyzed_c_operators_) {
	const auto& c_op = analyzed_c_operator.c_operator.get();

	for (const auto& i : analyzed_c_operator.state_with_nonzero_variable_collapse) {
	    gamma_from_i.try_emplace(i, 0);
	    gamma_from_i.at(i) += c_op.get_gamma_sum(density_matrix_.state_space_, i, t);
	}

	for (const auto& pair_i_jset : analyzed_c_operator.nonzero_collapse_pairs) {
	    const MyIndexType i = pair_i_jset.first;
	    const auto& jset = pair_i_jset.second;

	    const double rho_ii = std::real(rho_diag_part_row.values_ptr[i - rho_ownership_range_row_[0]]);
	    if (rho_ii == 0.0) continue;

	    // const std::unordered_map<MyIndexType, double>& map_j_gamma = analyzed_c_operator.cached_gamma.at(i);

	    for (const auto& j : jset) {
		if (rho_ii != 0.0) {
		    const double gamma = c_op.get_gamma(density_matrix_.state_space_, i, j, t).get_fill_gamma();
		    // const double gamma = map_j_gamma.at(j);
		    
		    const double fill_rate = rho_ii * gamma;

		    rate_to_j.try_emplace(j, 0.0);
		    rate_to_j.at(j) += fill_rate;
		}
	    }
	}

	for (const auto& pair_i_jset : analyzed_c_operator.nonzero_collapse_pairs_to_outer_state) {
	    const MyIndexType i = pair_i_jset.first;
	    const auto& jset = pair_i_jset.second;

	    const double rho_ii = std::real(rho_diag_part_row.values_ptr[i - rho_ownership_range_row_[0]]);
	    if (rho_ii == 0.0) continue;

	    for (const auto& j : jset) {
		const double gamma = c_op.get_gamma_to_outer_state(density_matrix_.state_space_, i, j, t).get_fill_gamma();
		const double fill_rate = rho_ii * gamma;

		rate_to_outer_j.try_emplace(j, 0.0);
		rate_to_outer_j.at(j) += fill_rate;
	    }
	}
    }


    // 
    // Initialize Drho_dt
    // 

    std::unordered_map<MyIndexType, std::unordered_set<MyIndexType>> Drho_Dt_nonzero_cols;

    for (const auto* mat : nonzero_place_source_for_Drho_Dt) {
	mat->loop_over_nonzero_elements(
		[&Drho_Dt_nonzero_cols, &Drho_Dt](const MyIndexType i, const MyIndexType j, const MyElementType element)
		{
		    Drho_Dt_nonzero_cols.try_emplace(i);
		    Drho_Dt_nonzero_cols.at(i).emplace(j);
		}
	    );
    }

    for (size_t rank = 0; rank < mpi_helper::get_size(); ++rank) {
	std::unordered_map<MyIndexType, double> rate_to_j_received;
	if (rank == mpi_helper::get_rank()) {
	    rate_to_j_received = rate_to_j;
	}
	boost::mpi::broadcast(mpi_helper::world, rate_to_j_received, rank);

	for (const auto& pair_j_rate : rate_to_j_received) {
	    const MyIndexType j = pair_j_rate.first;
	    if (rho_ownership_range_row_[0] <= j && j < rho_ownership_range_row_[1]) {
		Drho_Dt_nonzero_cols.try_emplace(j);
		Drho_Dt_nonzero_cols.at(j).emplace(j);
	    }
	}
    }

    MyMat::NonzeroNumbers Drho_Dt_number_of_nonzeros(rho_n_ownership_row_);
    for (MyIndexType i = rho_ownership_range_row_[0]; i < rho_ownership_range_row_[1]; ++i) {
	if (Drho_Dt_nonzero_cols.find(i) == Drho_Dt_nonzero_cols.end()) continue;

	const MyIndexType ii = i - rho_ownership_range_row_[0];

	for (const auto& j : Drho_Dt_nonzero_cols.at(i)) {
	    if (rho_ownership_range_row_[0] <= j && j < rho_ownership_range_row_[1]) ++Drho_Dt_number_of_nonzeros.diag[ii];
	    else ++Drho_Dt_number_of_nonzeros.nondiag[ii];
	}
    }

    Drho_Dt.set_preallocation(Drho_Dt_number_of_nonzeros);

    const MyElementType zero_value = 0.0;

    for (const auto& pair_i_jset : Drho_Dt_nonzero_cols) {
	const MyIndexType i = pair_i_jset.first;
	const auto& jset = pair_i_jset.second;

	for (const MyIndexType& j : jset) {
	    Drho_Dt.set_values(1, &i, 1, &j, &zero_value);
	}
    }

    Drho_Dt.begin_final_assembly();
    Drho_Dt.end_final_assembly();

    Dphi_Dt.set_all(0.0);


    //  
    // Calculate Liouvillian rank-by-rank
    //  
    for (size_t rank = 0; rank < mpi_helper::get_size(); ++rank) {
	std::unordered_map<MyIndexType, double> gamma_from_i_received;
	std::unordered_map<MyIndexType, double> rate_to_j_received;
	std::unordered_map<MyIndexType, double> rate_to_outer_j_received;
	if (rank == mpi_helper::get_rank()) {
	    gamma_from_i_received = std::move(gamma_from_i);
	    rate_to_j_received = std::move(rate_to_j);
	    rate_to_outer_j_received = std::move(rate_to_outer_j);
	}

	boost::mpi::broadcast(mpi_helper::world, gamma_from_i_received, rank);
	boost::mpi::broadcast(mpi_helper::world, rate_to_j_received, rank);
	boost::mpi::broadcast(mpi_helper::world, rate_to_outer_j_received, rank);

	auto rho_ownership_range_row = Drho_Dt.get_ownership_range_row();
	boost::mpi::broadcast(mpi_helper::world, rho_ownership_range_row, rank);

	orig_rho_.loop_by_row(
		[&Drho_Dt, &gamma_from_i_received](const MyIndexType i, const MyIndexType n_col, const MyIndexType* cols_ptr, const MyElementType* values_ptr)
		{
		    std::vector<MyIndexType> fill_cols;
		    std::vector<MyElementType> fill_values;

		    double gamma = 0;

		    if (gamma_from_i_received.find(i) != gamma_from_i_received.end()) {
			gamma += gamma_from_i_received.at(i);
		    }

		    for (MyIndexType jj = 0; jj < n_col; ++jj) {
			const MyIndexType j = cols_ptr[jj];
			if (gamma_from_i_received.find(j) != gamma_from_i_received.end()) {
			    gamma += gamma_from_i_received.at(j);
			}

			if (gamma != 0) {
			    fill_cols.push_back(j);
			    fill_values.push_back(-0.5 * gamma * values_ptr[jj]);
			}
		    }

		    Drho_Dt.add_values(1, &i, fill_cols.size(), &fill_cols[0], &fill_values[0]);
		}
	    );

	for (MyIndexType i = rho_ownership_range_row_[0]; i < rho_ownership_range_row_[1]; ++i) {
	    const MyIndexType j = i;
	    if (rate_to_j_received.find(j) != rate_to_j_received.end()) {
		MyElementType fill_value = rate_to_j_received.at(j);
		Drho_Dt.add_values(1, &i, 1, &j, &fill_value);
	    }
	}

	std::vector<MyIndexType> Dphi_Dt_fill_indices;
	std::vector<MyElementType> Dphi_Dt_fill_values;
	Dphi_Dt.loop_over_elements(
		[&Dphi_Dt, &rate_to_outer_j_received, &Dphi_Dt_fill_indices, &Dphi_Dt_fill_values](const MyIndexType i, const MyElementType value)
		{
		    const MyIndexType j = i;
		    if (rate_to_outer_j_received.find(j) != rate_to_outer_j_received.end()) {
			MyElementType fill_value = rate_to_outer_j_received.at(j);

			Dphi_Dt_fill_indices.push_back(j);
			Dphi_Dt_fill_values.push_back(fill_value);
		    }
		}
	    );
	Dphi_Dt.add_values(Dphi_Dt_fill_indices.size(), &Dphi_Dt_fill_indices[0], &Dphi_Dt_fill_values[0]);
    }

    Drho_Dt.begin_final_assembly();
    Drho_Dt.end_final_assembly();

    Dphi_Dt.begin_assembly();
    Dphi_Dt.end_assembly();

    rho_diag_part_vec.restore_row(rho_diag_part_row);
}


ErrorState abs(const OdeLindbladMasterMPI::State& x)
{
  ErrorState abs;
  abs.state_ = x;

  MyMat& abs_rho = abs.state_.get_rho();

  x.get_rho().loop_by_row(
      [&abs_rho](const MyIndexType i, const MyIndexType n_col, const MyIndexType* cols_ptr, const MyElementType* values_ptr) {
        std::vector<MyElementType> abs_rho_values(n_col);
        for (MyIndexType jj = 0; jj < n_col; ++jj) {
          abs_rho_values[jj] = std::abs(values_ptr[jj]);
        }
        abs_rho.set_values(1, &i, n_col, cols_ptr, &abs_rho_values[0]);
      }
    );

  abs_rho.begin_final_assembly();
  abs_rho.end_final_assembly();

  MyVec& abs_phi = abs.state_.get_phi();
  abs_phi.convert_to_abs();

  return abs;
}


State operator/(ErrorState&& x, ErrorState&& y)
{
  const MyMat& rho_x = x.state_.get_rho();
  const MyMat& rho_y = y.state_.get_rho();

  MyVec& phi_x = x.state_.get_phi();
  MyVec& phi_y = y.state_.get_phi();

  const double added_factor_to_x = x.added_factor_;
  const double added_factor_to_y = y.added_factor_;


  State divided;
  divided.instantiate(rho_x.get_n_dim(), phi_x.get_n_dim());

  MyMat& divided_rho = divided.get_rho();
  MyVec& divided_phi = divided.get_phi();


  MyMat::NonzeroNumbers nonzero_numbers_divided(divided_rho.get_n_ownership_row());

  {
      MyMat::NonzeroNumbers nonzero_numbers_x = rho_x.get_number_of_nonzeros_structured();
      MyMat::NonzeroNumbers nonzero_numbers_y = rho_y.get_number_of_nonzeros_structured();


      for (size_t i = 0; i < nonzero_numbers_divided.diag.size(); ++i) {
	  nonzero_numbers_divided.diag[i] = std::min(nonzero_numbers_x.diag[i] + nonzero_numbers_y.diag[i], divided_rho.get_n_ownership_row());
      }

      for (size_t i = 0; i < nonzero_numbers_divided.nondiag.size(); ++i) {
	  nonzero_numbers_divided.nondiag[i] = std::min(nonzero_numbers_x.nondiag[i] + nonzero_numbers_y.nondiag[i], divided_rho.get_n_dim());
      }
  }

  divided_rho.set_preallocation(nonzero_numbers_divided); // Not precise.

  rho_x.loop_by_row(
      [&added_factor_to_x, &rho_y, &added_factor_to_y, &divided_rho](const MyIndexType i, const MyIndexType n_col, const MyIndexType* cols_ptr, const MyElementType* values_ptr) {
        MyRow rho_row_y = rho_y.get_row(i);

        std::vector<MyIndexType> divided_indices;
	divided_indices.reserve(n_col + rho_row_y.n_col);

        std::vector<MyElementType> divided_values;
	divided_values.reserve(n_col + rho_row_y.n_col);

        MyIndexType jj_y = 0;

        for (MyIndexType jj_x = 0; jj_x < n_col; ++jj_x) {
          const MyIndexType j_x = cols_ptr[jj_x];

	  bool is_same_index_pair_found = false;

	  while (jj_y < rho_row_y.n_col && rho_row_y.cols_ptr[jj_y] <= j_x) {
	      MyElementType val_x = values_ptr[jj_x] + added_factor_to_x;

	      const MyIndexType j_y = rho_row_y.cols_ptr[jj_y];
	      const MyElementType val_y = added_factor_to_y + rho_row_y.values_ptr[jj_y];

	      if (j_y == j_x) {
		  val_x += values_ptr[jj_x];
		  is_same_index_pair_found = true;
	      }

	      divided_indices.push_back(j_y);
	      divided_values.push_back(val_x / val_y);

	      ++jj_y;
	  }

	  if (!is_same_index_pair_found) {
	      divided_indices.push_back(j_x);
	      divided_values.push_back((values_ptr[jj_x] + added_factor_to_x)/ added_factor_to_y);
	  }
        }

        divided_rho.set_values(1, &i, divided_indices.size(), &divided_indices[0], &divided_values[0]); 

        rho_y.restore_row(i, rho_row_y);
      }
    );

  divided_rho.begin_final_assembly();
  divided_rho.end_final_assembly();


  phi_x.add_to_all_elements(added_factor_to_x);
  phi_y.add_to_all_elements(added_factor_to_y);

  divided_phi = phi_x;
  divided_phi.pointwise_divide(phi_y);


  return divided;
}


} // namespace cppoqss
