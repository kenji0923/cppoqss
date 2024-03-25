#ifndef CPPOQSS_RESULT_ANALYZER_H
#define CPPOQSS_RESULT_ANALYZER_H


#include <cstddef>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>

#include <cppoqss/arithmetic.h>
#include <cppoqss/state_space.h>


namespace cppoqss {


class ResultAnalyzer
{
public:
    struct Element
    {
	Element(const MyIndexType j) : j(j), value(0.0) { }

	MyIndexType j;
	MyElementType value;
    };

    struct Row
    {
	Row(const MyIndexType i, const std::vector<Element>& elements) : i(i), elements(elements) { }

	MyIndexType i;
	std::vector<Element> elements;
    };

    struct MatQuery
    {
	double t;
	std::vector<std::vector<Row>> rows; 
    };

    struct VecQuery
    {
	double t;
	std::vector<Element> elements; 
    };

    ResultAnalyzer(const std::filesystem::path& read_path, const std::filesystem::path& solved_dir = "solved");

    const std::vector<double>& get_time_points() const { return time_points_; }

    size_t get_index(const double t_query) const;


    std::vector<size_t> get_rho_indices(const std::function<bool(const size_t)>& checker) const;

    void get_mat_element(MatQuery& query) const;

    void get_vec_element(VecQuery& query) const;

private:
    std::filesystem::path solved_path_;

    std::string state_type_;
    std::string ode_system_type_;

    std::shared_ptr<StateSpace> state_space_;

    std::vector<std::string> result_names_;

    std::vector<double> time_points_;
};


} // namespace cppoqss


#endif
