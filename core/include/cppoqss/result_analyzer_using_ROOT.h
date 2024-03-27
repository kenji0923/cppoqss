#ifndef CPPOQSS_RESULT_ANALYZER_USING_ROOT_H
#define CPPOQSS_RESULT_ANALYZER_USING_ROOT_H


#include <filesystem>
#include <vector>

#include <TGraph.h>

#include <cppoqss/result_analyzer.h>


namespace cppoqss {


class ResultAnalyzerUsingROOT
{
public:
  ResultAnalyzerUsingROOT(const std::filesystem::path& read_path, const std::filesystem::path& solved_dir = "solved");

  const std::vector<double>& get_time_points() const { return analyzer_.get_time_points(); }

  std::vector<TGraph*> get_time_evolution_graph(const std::vector<double>& time_points, const std::vector<ResultAnalyzer::Row>& rows) const;

  std::vector<TGraph*> get_phi_time_evolution_graph(const std::vector<double>& time_points, const std::vector<ResultAnalyzer::Element>& elements) const;

private:
  ResultAnalyzer analyzer_;
};


} // namespace cppoqss


#endif
