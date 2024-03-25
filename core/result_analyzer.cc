#include <cppoqss/result_analyzer.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_system_name.h>
#include <cppoqss/ode_system.h>


namespace cppoqss {


ResultAnalyzer::ResultAnalyzer(const std::filesystem::path& read_path, const std::filesystem::path& solved_dir)
: solved_path_(read_path / solved_dir)
{
    {
	std::ifstream ifs((read_path / "state_space.dat").c_str());

	boost::archive::text_iarchive ia(ifs);

	ia >> state_space_;
    }

    {
	std::ifstream ifs((read_path / "solver.dat").c_str());

	boost::archive::text_iarchive ia(ifs);

	ia >> state_type_;
	ia >> ode_system_type_;
    }

    if (ode_system_type_ == CPPOQSS_MACRO_SYSTEM_NAME_ODELINDBLAD) {
	auto results_saver = OdeLindbladMasterMPI::get_results_saver();

	for (const auto& saver : results_saver) {
	    result_names_.emplace_back(std::get<0>(saver));
	}
    } else {
	std::cerr << "unknown system type" << std::endl;
	exit(1);
    }

    printf("state space:\n");
    printf("  n_dim_rho: %zu\n", state_space_->get_n_dim_rho());
    printf("  n_dim_phi: %zu\n", state_space_->get_n_dim_phi());

    std::cout << "results: " << std::flush;
    for (const auto& name : result_names_) {
	std::cout << name << " " << std::flush;
    }
    std::cout << std::endl;
}


size_t ResultAnalyzer::get_index(const double t_query) const
{
    auto it_lower_bound = std::lower_bound(time_points_.begin(), time_points_.end(), t_query);

    if (it_lower_bound == time_points_.begin()) return 0;
    if (it_lower_bound == time_points_.end()) return time_points_.size() - 1;

    const double t_early = *(it_lower_bound - 1);
    const double t_later = *(it_lower_bound);

    const size_t distance = std::distance(time_points_.begin(), it_lower_bound);

    if (t_query - t_early < t_later - t_query) {
	return distance - 1;
    } else {
	return distance;
    }
}


std::vector<size_t> ResultAnalyzer::get_rho_indices(const std::function<bool(const size_t)>& checker) const
{
    std::vector<size_t> indices;

    for (size_t i = 0; i < state_space_->get_n_dim_rho(); ++i) {
	if (checker(i)) indices.push_back(i);
    }

    return indices;
}


void ResultAnalyzer::get_mat_element(MatQuery& query) const
{
    // const size_t index_query = result_.get_index(query.t);
 //    query.t = result_.get_result().time_point.data[index_query];
	// 
 //    std::unique_ptr<IResultState> rho = result_.get_rho(index_query);
	// 
 //    for (auto& row : query.rows) {
	// const MyIndexType i = row.i;
	// 
	// MyRow result_row = rho->get_row(i);
	// for (auto& element : row.elements) {
	//     const MyIndexType j = element.j;
	//     MyElementType& value = element.value;
	// 
	//     auto j_it = std::lower_bound(result_row.cols_ptr, result_row.cols_ptr + result_row.n_col, j);
	//     if (j_it != result_row.cols_ptr + result_row.n_col && *j_it == j) {
	// 	value = result_row.values_ptr[std::distance(result_row.cols_ptr, j_it)];
	//     }
	// }
	// rho->restore_row(i, result_row);
 //    }
}


void ResultAnalyzer::get_vec_element(VecQuery& query) const
{
 //    const size_t index_query = result_.get_index(query.t);
 //    query.t = result_.get_result().time_point.data[index_query];
	// 
 //    std::unique_ptr<MyVec> phi = result_.get_phi(index_query);
	// 
 //    MyRow result_row = phi->get_row();
 //    for (auto& element : query.elements) {
	// const MyIndexType j = element.j;
	// MyElementType& value = element.value;
	// 
	// auto j_it = std::lower_bound(result_row.cols_ptr, result_row.cols_ptr + result_row.n_col, j);
	// if (j_it != result_row.cols_ptr + result_row.n_col && *j_it == j) {
	//     value = result_row.values_ptr[std::distance(result_row.cols_ptr, j_it)];
	// }
 //    }
 //    phi->restore_row(result_row);
}


} // namespace cppoqss
