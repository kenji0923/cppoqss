#include <cppoqss/result_analyzer_using_ROOT.h>

#include <vector>

#include "TGraph.h"

#include <cppoqss/arithmetic.h>
#include <cppoqss/progress_bar.h>
#include <cppoqss/result_analyzer.h>


namespace cppoqss {


ResultAnalyzerUsingROOT::ResultAnalyzerUsingROOT(const std::filesystem::path& read_path, const std::filesystem::path& solved_dir)
: analyzer_(read_path, solved_dir)
{ }


std::vector<TGraph*> ResultAnalyzerUsingROOT::get_time_evolution_graph(const std::vector<double>& time_points, const std::vector<ResultAnalyzer::Row>& rows) const
{
  size_t n_element = 0;
  for (const auto& row : rows) {
    n_element += row.elements.size();
  }

  std::vector<TGraph*> graph_list(n_element);
  // {
  //   size_t i_graph = 0;
  //   for (const auto& row : rows) {
  //     const MyIndexType i = row.i;
  //     for (const auto& element : row.elements) {
  //       const MyIndexType j = element.j;
		// 
  //       graph_list[i_graph] = new TGraph(time_points.size());
  //       TGraph*& graph = graph_list[i_graph];
		// 
  //       graph->SetNameTitle(Form("g_rho_%zu_%zu", static_cast<size_t>(i), static_cast<size_t>(j)), Form("time-evolution of rho_%zu_%zu", static_cast<size_t>(i), static_cast<size_t>(j)));
		// 
  //       ++i_graph;
  //     }
  //   }
  // }
		// 
  // ProgressBar::InitBar("Analyze result", time_points.size());
  // for (size_t i_time = 0; i_time < time_points.size(); ++i_time) {
  //   const double t = time_points[i_time];
		// 
  //   ResultAnalyzer::MatQuery query { t, rows };
		// 
  //   analyzer_.get_mat_element(query);
		// 
  //   {
  //     size_t i_graph = 0;
  //     for (const auto& row : query.rows) {
  //       for (const auto& element : row.elements) {
  //         graph_list[i_graph]->SetPoint(i_time, query.t, std::abs(element.value));
  //         ++i_graph;
  //       }
  //     }
  //   }
		// 
  //   ProgressBar::ProgressStep();
  // }

  return graph_list;
}


std::vector<TGraph*> ResultAnalyzerUsingROOT::get_phi_time_evolution_graph(const std::vector<double>& time_points, const std::vector<ResultAnalyzer::Element>& elements) const
{
  size_t n_element = 0;
  for (const auto& element : elements) {
    n_element += 1;
  }

  std::vector<TGraph*> graph_list(n_element);
  // {
  //   size_t i_graph = 0;
  //   for (const auto& element : elements) {
  //     const MyIndexType j = element.j;
		// 
  //     graph_list[i_graph] = new TGraph(time_points.size());
  //     TGraph*& graph = graph_list[i_graph];
		// 
  //     graph->SetNameTitle(Form("g_phi_%zu", static_cast<size_t>(j)), Form("time-evolution of phi_%zu", static_cast<size_t>(j)));
		// 
  //     ++i_graph;
  //   }
  // }
		// 
  // ProgressBar::InitBar("Analyze result", time_points.size());
  // for (size_t i_time = 0; i_time < time_points.size(); ++i_time) {
  //   const double t = time_points[i_time];
		// 
  //   ResultAnalyzer::VecQuery query { t, elements };
		// 
  //   analyzer_.get_vec_element(query);
		// 
  //   {
  //     size_t i_graph = 0;
  //     for (const auto& element : query.elements) {
  //       graph_list[i_graph]->SetPoint(i_time, query.t, std::abs(element.value));
  //       ++i_graph;
  //     }
  //   }
		// 
  //   ProgressBar::ProgressStep();
  // }

  return graph_list;
}


} // namespace cppoqss
