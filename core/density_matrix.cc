#include <cppoqss/density_matrix.h>

#include <cassert>
#include <complex>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/mpi.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_space.h>
#include <cppoqss/mpi_helper.h>
#include <cppoqss/progress_bar.h>


namespace cppoqss {


const std::string DensityMatrix::type = "DensityMatrix";


DensityMatrix::DensityMatrix(
	const std::shared_ptr<const StateSpace>& state_space,
	std::function<MyElementType(const StateSpace&, const MyIndexType, const MyIndexType)> rho_element_getter
	)
    : ptr_state_space_(state_space), n_dim_rho_(state_space->get_n_dim_rho()), n_dim_phi_(state_space->get_n_dim_phi()), rho_(n_dim_rho_), phi_(n_dim_phi_), t_(0.0), state_space_(*state_space), const_rho_(rho_), const_phi_(phi_)
{
    //
    // Start initializing rho.
    // 
    const auto& row_ownership_range = rho_.get_ownership_range_row();
    const MyIndexType n_row_ownership = rho_.get_n_ownership_row();

    std::vector<std::vector<MyIndexType>> nonzero_col(n_row_ownership);
    std::vector<std::vector<MyElementType>> nonzero_element(n_row_ownership);

    double diagonal_sum = 0;

    const MyIndexType ownership_start_row = row_ownership_range[0];

    ProgressBar::InitBar("initialize density matrix", n_row_ownership);
    MyIndexType current_row = -1;

    auto nonzero_checker = [&state_space, &rho_element_getter, &ownership_start_row, &nonzero_col, &nonzero_element, &diagonal_sum, &current_row](const MyIndexType i, const MyIndexType j) {
	const MyIndexType bra_sindex = i;
	const MyIndexType ket_sindex = j;

	const MyElementType element = rho_element_getter(*state_space, bra_sindex, ket_sindex);

	if (i == j) {
	    if (std::imag(element) != 0.0) {
		std::cerr << "Diagonal part is not real." << std::endl;
		exit(1);
	    }
	    diagonal_sum += std::real(element);
	}

	if (current_row != i) {
	    if (current_row >= 0) ProgressBar::ProgressStep();
	    current_row = i;
	}

	if (element == MyElementType(0)) {
	    return false;
	} else {
	    const MyIndexType ii = i - ownership_start_row;
	    nonzero_col[ii].emplace_back(j);
	    nonzero_element[ii].emplace_back(element);
	    return true;
	}
    };

    const auto nonzero_numbers = rho_.get_nonzero_numbers(nonzero_checker);
    ProgressBar::ProgressStep();

    diagonal_sum = boost::mpi::all_reduce(mpi_helper::world, diagonal_sum, std::plus<double>());

    rho_.set_preallocation(0, &nonzero_numbers.diag[0], 0, &nonzero_numbers.nondiag[0]);

    const MyIndexType n_nonzero_diag = std::accumulate(nonzero_numbers.diag.begin(), nonzero_numbers.diag.end(), 0);
    const MyIndexType n_nonzero_nondiag = std::accumulate(nonzero_numbers.nondiag.begin(), nonzero_numbers.nondiag.end(), 0);

    const MyIndexType n_nonzero_diag_sum = boost::mpi::all_reduce(mpi_helper::world, n_nonzero_diag, std::plus<MyIndexType>());
    const MyIndexType n_nonzero_nondiag_sum = boost::mpi::all_reduce(mpi_helper::world, n_nonzero_nondiag, std::plus<MyIndexType>());

    for (MyIndexType ii = 0; ii < n_row_ownership; ++ii) {
	auto& elements = nonzero_element[ii];
	for (auto& element : elements) {
	    element /= diagonal_sum;
	}
	const MyIndexType i = row_ownership_range[0] + ii;
	rho_.set_values(1, &i, nonzero_col[ii].size(), &nonzero_col[ii][0], &nonzero_element[ii][0]);
    }

    rho_.begin_final_assembly();
    rho_.end_final_assembly();


    //  
    // Start initializing phi.
    //  
    phi_.set_all(0);
    phi_.begin_assembly();
    phi_.end_assembly();


    //
    // Print results.
    //
    if (mpi_helper::is_printing_rank()) {
	printf("density matrix is initialized\n");
	printf("non-zero diagonal: %ld elements\n", n_nonzero_diag_sum);
	printf("non-zero off-diagonal: %ld elements\n", n_nonzero_nondiag_sum);
    }
}


} // namespace cppoqss
