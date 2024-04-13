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
#include <boost/serialization/array.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/c_operator.h>
#include <cppoqss/density_matrix.h>
#include <cppoqss/logger.h>
#include <cppoqss/mpi_helper.h>
#include <cppoqss/operator.h>
#include <cppoqss/progress_bar.h>
#include <cppoqss/state_space.h>
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
:   density_matrix_(density_matrix),
    hamiltonian_(density_matrix_.state_space_.get_hamiltonian()),
    Hmat_(density_matrix_.get_n_dim_rho(), true),
    constant_fill_by_collapse_to_rho_(density_matrix_.get_n_dim_rho()),
    constant_fill_by_collapse_to_phi_(density_matrix_.get_n_dim_phi(), density_matrix_.get_n_dim_rho()),
    // TODO implement
    // variable_collapse_gamma_(density_matrix_.get_n_dim_rho()),
    // variable_fill_to_rho_(density_matrix_.get_n_dim_rho()),
    // variable_fill_to_phi_(density_matrix_.get_n_dim_phi(), density_matrix_.get_n_dim_rho()),
    orig_rho_(nullptr),
    t0_for_multiply_collapse_prob_(density_matrix.get_t())
{ 
    if (mpi_helper::is_printing_rank()) {
	printf("initialize ODE system\n");
    }

    // 
    // Prepare Hamiltonian matrix
    // 
    rho_n_dim_ = density_matrix_.get_n_dim_rho();
    rho_ownership_range_row_ = density_matrix_.get_rho().get_ownership_range_row();
    rho_n_ownership_row_ = rho_ownership_range_row_[1] - rho_ownership_range_row_[0];
    boost::mpi::all_gather(mpi_helper::world, rho_ownership_range_row_, rho_ownership_range_row_of_each_rank_);

    phi_n_dim_ = density_matrix_.get_n_dim_phi();
    phi_ownership_range_ = density_matrix_.get_phi().get_ownership_range();
    phi_n_ownership_ = phi_ownership_range_[1] - phi_ownership_range_[0];
    boost::mpi::all_gather(mpi_helper::world, phi_ownership_range_, phi_ownership_range_of_each_rank_);

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

    Hmat_.set_preallocation(0, &Hmat_nonzero_numbers.diag[0], 0, &Hmat_nonzero_numbers.nondiag[0]);

    evaluate_hamiltonian(0);

    const MyIndexType Hmat_number_of_nonzeros = Hmat_.get_number_of_nonzeros();

    if (mpi_helper::is_printing_rank()) {
	printf("Hamiltonian non-zeros: %ld\n", Hmat_number_of_nonzeros);
    }


    //
    // Analyze collapse
    //

    SparseMatrixNonzeroStruct nonzero_struct_constant_fill_ro_rho;
    SparseMatrixNonzeroStruct nonzero_struct_constant_fill_ro_phi;
    // SparseMatrixNonzeroStruct variable_fill_to_rho_nonzero_struct;
    // SparseMatrixNonzeroStruct variable_fill_to_phi_nonzero_struct;

    ProgressBar::InitBar("analyze c-operator", rho_n_ownership_row_ + 2 * (rho_n_ownership_row_ + phi_n_ownership_));

    for (MyIndexType i = rho_ownership_range_row_[0]; i < rho_ownership_range_row_[1]; ++i) {
	double gamma_sum = 0.0;

	for (const ICOperator& c_op : density_matrix_.state_space_.get_c_operators()) {
	    if (!c_op.is_constant()) continue;

	    gamma_sum += c_op.get_gamma_sum(density_matrix_.state_space_, i, 0);
	}

	if (gamma_sum != 0.0) {
	    constant_collapse_gamma_sum_[i] = gamma_sum;
	}

	ProgressBar::ProgressStep();
    }

    current_row = -1;

    constant_fill_by_collapse_to_rho_.set_preallocation(
	    constant_fill_by_collapse_to_rho_.get_nonzero_numbers(
		    [this, &nonzero_struct_constant_fill_ro_rho, &current_row](const MyIndexType i, const MyIndexType j)
		    {
			const MyIndexType source = j;
			const MyIndexType target = i;

			const MyIndexType ii = i - rho_ownership_range_row_[0];

			if (nonzero_struct_constant_fill_ro_rho.get_n_row() <= ii) {
			    nonzero_struct_constant_fill_ro_rho.push_back_row();
			}

			bool is_nonzero = false;

			for (const auto& c_operator_ref : this->density_matrix_.state_space_.get_c_operators()) {
			    const ICOperator& c_op = c_operator_ref.get();

			    if (!c_op.is_constant()) continue;

			    is_nonzero = is_nonzero || c_op.is_gamma_nonzero(this->density_matrix_.state_space_, source, target);
			}

			if (is_nonzero) {
			    nonzero_struct_constant_fill_ro_rho.push_back_element(source);
			}

			if (current_row != i) {
			    if (current_row >= 0) ProgressBar::ProgressStep();
			    current_row = i;
			}

			return is_nonzero;
		    }
		)
	    );
    
    nonzero_struct_constant_fill_ro_rho.push_back_row();

    current_row = -1;

    nonzero_struct_constant_fill_ro_rho.loop_by_row(
	    [this, &current_row](const MyIndexType ii, const MyIndexType n_element, const MyIndexType* cols)
	    {
		ProgressBar::ProgressStep();

		if (n_element == 0) return;

		const MyIndexType i = rho_ownership_range_row_[0] + ii;

		const MyIndexType target = i;

		std::vector<MyElementType> elements(n_element, 0.0);

		for (MyIndexType i_element = 0; i_element < n_element; ++i_element) {
		    const MyIndexType j = cols[i_element];

		    const MyIndexType source = j;

		    for (const auto& c_operator_ref : this->density_matrix_.state_space_.get_c_operators()) {
			const ICOperator& c_op = c_operator_ref.get();

			if (!c_op.is_constant()) continue;

			elements[i_element] += c_op.get_gamma(density_matrix_.state_space_, source, target, 0).get_fill_gamma();
		    }
		}

		constant_fill_by_collapse_to_rho_.set_values(1, &i, n_element, cols, &elements[0]);
	    }
	);

    constant_fill_by_collapse_to_rho_.begin_final_assembly();


    current_row = -1;

    constant_fill_by_collapse_to_phi_.set_preallocation(
	    constant_fill_by_collapse_to_phi_.get_nonzero_numbers(
		    [this, &nonzero_struct_constant_fill_ro_phi, &current_row](const MyIndexType i, const MyIndexType j)
		    {
			const MyIndexType source = j;
			const MyIndexType target = i;

			const MyIndexType ii = i - phi_ownership_range_[0];

			if (nonzero_struct_constant_fill_ro_phi.get_n_row() <= ii) {
			    nonzero_struct_constant_fill_ro_phi.push_back_row();
			}

			bool is_nonzero = false;

			for (const auto& c_operator_ref : this->density_matrix_.state_space_.get_c_operators()) {
			    const ICOperator& c_op = c_operator_ref.get();

			    if (!c_op.is_constant()) continue;

			    is_nonzero = is_nonzero || c_op.is_gamma_to_outer_state_nonzero(this->density_matrix_.state_space_, source, target);
			}

			if (is_nonzero) {
			    nonzero_struct_constant_fill_ro_phi.push_back_element(source);
			}

			if (current_row != i) {
			    if (current_row >= 0) ProgressBar::ProgressStep();
			    current_row = i;
			}

			return is_nonzero;
		    }
		)
	    );

    nonzero_struct_constant_fill_ro_phi.push_back_row();

    current_row = -1;

    nonzero_struct_constant_fill_ro_phi.loop_by_row(
	    [this, &current_row](const MyIndexType ii, const MyIndexType n_element, const MyIndexType* cols)
	    {
		ProgressBar::ProgressStep();

		if (n_element == 0) return;

		const MyIndexType i = phi_ownership_range_[0] + ii;

		const MyIndexType target = i;

		std::vector<MyElementType> elements(n_element, 0.0);

		for (MyIndexType i_element = 0; i_element < n_element; ++i_element) {
		    const MyIndexType j = cols[i_element];

		    const MyIndexType source = j;

		    for (const auto& c_operator_ref : this->density_matrix_.state_space_.get_c_operators()) {
			const ICOperator& c_op = c_operator_ref.get();

			if (!c_op.is_constant()) continue;

			elements[i_element] += c_op.get_gamma_to_outer_state(density_matrix_.state_space_, source, target, 0).get_fill_gamma();
		    }
		}

		constant_fill_by_collapse_to_phi_.set_values(1, &i, n_element, cols, &elements[0]);
	    }
	);

    constant_fill_by_collapse_to_phi_.begin_final_assembly();


    constant_fill_by_collapse_to_rho_.end_final_assembly();
    constant_fill_by_collapse_to_phi_.end_final_assembly();
    

    // TODO analyze variable collapse

    size_t n_collapse_pair = constant_fill_by_collapse_to_rho_.get_number_of_nonzeros() + constant_fill_by_collapse_to_phi_.get_number_of_nonzeros();

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
		if (n_element == 0) return;

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

  // TODO avoid copy
  std::unordered_map<MyIndexType, double> constant_gamma = constant_collapse_gamma_sum_;

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
    MyVec rho_diag_part_vec = orig_rho_.get_diagonal();

    MyVec constant_fill_prob_to_rho = constant_fill_by_collapse_to_rho_.multiply(rho_diag_part_vec);

    MyVec constant_fill_prob_to_phi = constant_fill_by_collapse_to_phi_.multiply(rho_diag_part_vec);


    // 
    // Initialize Drho_dt
    // 

    std::unordered_map<MyIndexType, std::unordered_set<MyIndexType>> Drho_Dt_nonzero_cols;

    for (const auto* mat : nonzero_place_source_for_Drho_Dt) {
	mat->loop_over_nonzero_elements(
		[&Drho_Dt_nonzero_cols, &Drho_Dt](const MyIndexType i, const MyIndexType j, const MyElementType element)
		{
		    Drho_Dt_nonzero_cols[i].emplace(j);
		}
	    );
    }

    constant_fill_prob_to_rho.loop_over_elements(
	    [&](const MyIndexType i, const MyElementType element)
	    {
		if (element != 0.0) {
		    Drho_Dt_nonzero_cols[i].emplace(i);
		}
	    }
	);

    MyMat::NonzeroNumbers Drho_Dt_number_of_nonzeros(rho_n_ownership_row_);

    for (MyIndexType i = rho_ownership_range_row_[0]; i < rho_ownership_range_row_[1]; ++i) {
	if (Drho_Dt_nonzero_cols.find(i) == Drho_Dt_nonzero_cols.end()) continue;

	const MyIndexType ii = i - rho_ownership_range_row_[0];

	for (const auto& j : Drho_Dt_nonzero_cols[i]) {
	    if (rho_ownership_range_row_[0] <= j && j < rho_ownership_range_row_[1]) ++Drho_Dt_number_of_nonzeros.diag[ii];
	    else ++Drho_Dt_number_of_nonzeros.nondiag[ii];
	}
    }

    Drho_Dt.set_preallocation(Drho_Dt_number_of_nonzeros);

    const MyElementType zero_value = 0.0;

    for (const auto& [i, jset] : Drho_Dt_nonzero_cols) {
	for (const MyIndexType& j : jset) {
	    Drho_Dt.set_values(1, &i, 1, &j, &zero_value);
	}
    }

    Drho_Dt.begin_flush_assembly();
    Drho_Dt.end_flush_assembly();

    Dphi_Dt.set_all(0.0);


    //  
    // Calculate Liouvillian rank-by-rank
    //  
    
    // TODO implement variable collapse sum
 //    for (size_t i_rank = 0; i_rank < mpi_helper::get_size(); ++i_rank) {
	// std::unordered_map<MyIndexType, double> gamma_from_i_received;
	// 
	// if (i_rank == mpi_helper::get_rank()) {
	//     gamma_from_i_received = std::move(gamma_from_i);
	// }
	// 
	// boost::mpi::broadcast(mpi_helper::world, gamma_from_i_received, i_rank);
	// 
	// auto rho_ownership_range_row = Drho_Dt.get_ownership_range_row();
	// boost::mpi::broadcast(mpi_helper::world, rho_ownership_range_row, i_rank);
	// 
	// orig_rho_.loop_by_row(
	// 	[&Drho_Dt, &gamma_from_i_received](const MyIndexType i, const MyIndexType n_col, const MyIndexType* cols_ptr, const MyElementType* values_ptr)
	// 	{
	// 	    if (n_col == 0) return;
	// 
	// 	    std::vector<MyIndexType> fill_cols;
	// 	    std::vector<MyElementType> fill_values;
	// 
	// 	    double gamma = 0;
	// 
	// 	    if (gamma_from_i_received.find(i) != gamma_from_i_received.end()) {
	// 		gamma += gamma_from_i_received.at(i);
	// 	    }
	// 
	// 	    for (MyIndexType jj = 0; jj < n_col; ++jj) {
	// 		const MyIndexType j = cols_ptr[jj];
	// 		if (gamma_from_i_received.find(j) != gamma_from_i_received.end()) {
	// 		    gamma += gamma_from_i_received.at(j);
	// 		}
	// 
	// 		if (gamma != 0) {
	// 		    fill_cols.push_back(j);
	// 		    fill_values.push_back(-0.5 * gamma * values_ptr[jj]);
	// 		}
	// 	    }
	// 
	// 	    Drho_Dt.add_values(1, &i, fill_cols.size(), &fill_cols[0], &fill_values[0]);
	// 	}
	//     );
	// 
 //    }

    constant_fill_prob_to_rho.loop_over_elements(
	    [&Drho_Dt](const MyIndexType i, const MyElementType element)
	    {
		if (element != 0.0) {
		    Drho_Dt.add_values(1, &i, 1, &i, &element);
		}
	    }
	);

    Drho_Dt.begin_final_assembly();

    constant_fill_prob_to_phi.loop_over_elements(
	    [&Dphi_Dt](const MyIndexType i, const MyElementType element)
	    {
		Dphi_Dt.add_values(1, &i, &element);
	    }
	);

    Dphi_Dt.begin_assembly();

    Drho_Dt.end_final_assembly();
    Dphi_Dt.end_assembly();
}


ErrorState abs(const OdeLindbladMasterMPI::State& x)
{
  ErrorState abs;
  abs.state_ = x;

  MyMat& abs_rho = abs.state_.get_rho();

  x.get_rho().loop_by_row(
      [&abs_rho](const MyIndexType i, const MyIndexType n_col, const MyIndexType* cols_ptr, const MyElementType* values_ptr) {
	if (n_col == 0) return;

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
