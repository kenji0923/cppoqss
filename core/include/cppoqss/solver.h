#ifndef CPPOQSS_SOLVER_H
#define CPPOQSS_SOLVER_H


#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <boost/archive/text_oarchive.hpp>
#include <boost/format.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>

#include <cppoqss/c_operator.h>
#include <cppoqss/density_matrix.h>
#include <cppoqss/logger.h>
#include <cppoqss/ode_integrator_function.h>
#include <cppoqss/operator.h>
#include <cppoqss/progress_bar.h>
#include <cppoqss/result.h>
#include <cppoqss/unit.h>


namespace cppoqss {


template<class StepperType>
struct SolverStepperWrapperFixed
{
    SolverStepperWrapperFixed(StepperType&& stepper) : stepper(std::forward<StepperType>(stepper)) { }

    std::string dump() const { return type; }

    StepperType stepper;
    const std::string type = "fixed";
};


template<class StepperType>
struct SolverStepperWrapperAdaptive
{
    SolverStepperWrapperAdaptive(StepperType&& stepper, const double eps_rel, const double eps_abs) : stepper(std::forward<StepperType>(stepper)), eps_rel(eps_rel), eps_abs(eps_abs) { }

    std::string dump() const { return (boost::format("%s\neps_rel:%.17g\neps_abs:%.17g") % type % eps_rel % eps_abs).str(); }

    StepperType stepper;
    const std::string type = "adaptive";
    const double eps_rel;
    const double eps_abs;
};


template<class StateType, class SystemType>
class Solver
{
public:
    class ObservationPointList
    {
    public:
	double get_t(const size_t i) const { return points_.at(i); }
	double get_save_mode(const size_t i) const { return save_mode_.at(i); }
	double get_step(const size_t i) const { return points_[i + 1] - points_[i]; }

	int get_save_mode(const double t) const;

	void push_back_point(const double t, const int mode);
	void set_start_point(const double t0);

	auto begin() { return points_.begin(); }
	auto begin() const { return points_.begin(); }
	auto end() { return points_.end(); }
	auto end() const { return points_.end(); }

	auto front() const { return points_.front(); }
	auto back() const { return points_.back(); }

    private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version);

	std::vector<double> points_; /**< Must be sorted. */
	std::vector<int> save_mode_; /**< If mode is negative, saving process is ignored. The mode 0 is for saving full data. */
    };

    class Observer
    {
    public:
	Observer(SystemType& ode_system, const ObservationPointList& obs_points, const std::filesystem::path& dir_result_saved, const unsigned int force_saving_interval);

	void operator()(const typename SystemType::State& x, const double t);
	void record_time_step(const double t, const double dt);
	void record_failed_time_step(const double t, const double dt_tried, const double dt_new);

    private:
	void update_progress_bar(const char* time_str, const double t) const;

	SystemType& ode_system_;
	const ObservationPointList& obs_points_;
	const std::filesystem::path& dir_result_saved_;
	unsigned int force_saving_interval_;

	int n_div_for_progress_bar_;

	ResultData<std::vector<double>> result_time_points; /**< [ (t, dt) ] */
	ResultData<std::vector<std::array<double, 2>>> result_time_steps; /**< [ (t, dt) ] */
	ResultData<std::vector<std::array<double, 3>>> result_failed_time_steps; /**< [ (t, dt_tried, dt_new) ] */
	std::vector<ResultMeta<typename SystemType::State>> results_;

	mutable size_t current_index_of_obs_points_ = 0;
	mutable std::chrono::high_resolution_clock::time_point timer_last_saved_;
    };

    Solver(StateType& state, const std::string& result_filename, const unsigned int force_saving_interval);

    ObservationPointList get_obspoints_constant_interval(const double t_end, const double dt, const unsigned int steps_per_save) const;

    auto get_constant_stepper() const;
    auto get_adaptive_stepper(const double eps_rel, const double eps_abs=DBL_MAX) const;

    template<class StepperType>
    size_t solve(ObservationPointList& obs_points, StepperType& stepper);

private:
    double get_default_eps_abs(const double eps_rel, const double eps_abs) const;

    SystemType ode_system_;
    std::string result_filename_;
    unsigned int force_saving_interval_;
    std::filesystem::path save_dir_;
};


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::ObservationPointList::push_back_point(const double t, const int mode)
{
    points_.push_back(t);
    save_mode_.push_back(mode);
}


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::ObservationPointList::set_start_point(const double t0)
{
    auto it_later_than_t0 = std::lower_bound(points_.begin(), points_.end(), t0);

    std::vector<double> new_points { t0 };
    std::vector<int> new_save_mode { 0 };

    for (auto it = it_later_than_t0; it != points_.end(); ++it) {
	if (*it > new_points.back()) {
	    size_t distance = std::distance(points_.begin(), it);

	    new_points.push_back(points_[distance]);
	    new_save_mode.push_back(save_mode_[distance]);
	}
    }

    points_ = std::move(new_points);
    save_mode_ = std::move(new_save_mode);
}


template<class StateType, class SystemType>
template<class Archive>
void Solver<StateType, SystemType>::ObservationPointList::serialize(Archive& ar, const unsigned int version)
{
    ar & points_;
    ar & save_mode_;
}


template<class StateType, class SystemType>
Solver<StateType, SystemType>::Observer::Observer(SystemType& ode_system, const ObservationPointList& obs_points, const std::filesystem::path& dir_result_saved, const unsigned int force_saving_interval)
:   ode_system_(ode_system),
    obs_points_(obs_points),
    dir_result_saved_(dir_result_saved),
    force_saving_interval_(force_saving_interval),
    n_div_for_progress_bar_(100),
    result_time_points("time_points.dat"),
    result_time_steps("time_steps.dat"),
    result_failed_time_steps("failed_time_steps.dat")

{
    ProgressBar::InitBar("Solving time-evolution", n_div_for_progress_bar_);
    ProgressBar::print_progress();

    for (const auto& name_and_savers : SystemType::get_results_saver()) {
	results_.emplace_back(dir_result_saved_, std::get<0>(name_and_savers), std::get<1>(name_and_savers), std::get<2>(name_and_savers));
    }
}


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::Observer::operator()(const typename SystemType::State& x, const double t)
{
    //
    // This should never happen.
    //
    if (t != obs_points_.get_t(current_index_of_obs_points_)) {
	fprintf(stderr, "not correctly observed\n");
	exit(1);
    }

    result_time_points.data.emplace_back(t);

    ode_system_.pre_observation_process(const_cast<typename SystemType::State&>(x), t);

    const int save_mode = obs_points_.get_save_mode(current_index_of_obs_points_);

    const auto timer_now = std::chrono::high_resolution_clock::now();
    const auto timer_elapsed = std::chrono::duration_cast<std::chrono::seconds>(timer_now - timer_last_saved_);

    bool is_save_interval_exceeded;

    if (mpi_helper::is_manager_rank()) {
	is_save_interval_exceeded= timer_elapsed.count() > force_saving_interval_;
    }

    boost::mpi::broadcast(mpi_helper::world, is_save_interval_exceeded, mpi_helper::get_manager_rank());

    if (save_mode >= 0) {
	result_time_points.save(dir_result_saved_);
	result_time_steps.save(dir_result_saved_);
	result_failed_time_steps.save(dir_result_saved_);

	for (auto& result : results_) {
	    result.append(x, save_mode);
	}
    }

    if (save_mode != 0 && is_save_interval_exceeded) {
	const std::filesystem::path& tmp_directory = dir_result_saved_ / "tmp/";

	if (mpi_helper::is_manager_rank()) {
	    std::ofstream ofs((tmp_directory / "time_point.dat").c_str());
	    boost::archive::text_oarchive oa(ofs);
	    oa << t;
	}

	for (auto& result : results_) {
	    result.rewrite_data_mode0(x, tmp_directory);
	}
    }

    if (mpi_helper::is_printing_rank()) {
	char time_str[128];
	sprintf(time_str, "t=%3.2ens                                       ", t / u::ns);
	update_progress_bar(time_str, t);
    }

    ode_system_.post_observation_process(const_cast<typename SystemType::State&>(x), t);

    ++current_index_of_obs_points_;
}


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::Observer::record_time_step(const double t, const double dt)
{
    result_time_steps.data.emplace_back(std::array<double, 2> { t, dt });

    if (mpi_helper::is_printing_rank()) {
	char time_str[128];
	sprintf(time_str, "t=%3.2ens dt=%2.1eps                           ", t/u::ns, dt/u::ps);
	update_progress_bar(time_str, t);
    }
}


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::Observer::record_failed_time_step(const double t, const double dt_tried, const double dt_new)
{
    result_failed_time_steps.data.emplace_back(std::array<double, 3> { t, dt_tried, dt_new });

    if (mpi_helper::is_printing_rank()) {
	char time_str[128];
	sprintf(time_str, "t=%3.2ens dt=%2.1eps->%2.1eps at t=%3.2ens", t/u::ns, dt_tried/u::ps, dt_new/u::ps, t/u::ns);
	update_progress_bar(time_str, t);
    }
}


template<class StateType, class SystemType>
void Solver<StateType, SystemType>::Observer::update_progress_bar(const char* time_str, const double t) const
{
    ProgressBar::set_option(ProgressBar::PostfixText{ time_str });
    ProgressBar::set_progress(std::floor((t - obs_points_.front()) / obs_points_.back() * n_div_for_progress_bar_));
}


template<class StateType, class SystemType>
Solver<StateType, SystemType>::Solver(StateType& state, const std::string& result_filename, const unsigned int force_saving_interval)
: ode_system_(state), result_filename_(result_filename), force_saving_interval_(force_saving_interval)
{
    std::filesystem::path result_base_dir("result");

    std::filesystem::path save_dir_title = result_filename_;

    if (mpi_helper::is_manager_rank()) {
	std::filesystem::create_directories(result_base_dir);

	std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss_timestamp;
	ss_timestamp << std::put_time(localtime(&now_c), "%Y-%m-%d-%H%M%S");

	std::string result_dir_name = std::string(save_dir_title.c_str()) + "_" + ss_timestamp.str();
	save_dir_ = result_base_dir / result_dir_name;

	if (std::filesystem::is_directory(save_dir_)) {
	    std::cerr << "conflicting result dir \"" << save_dir_.c_str() << "\" found" << std::endl;
	    exit(1);
	}
	std::filesystem::create_directory(save_dir_);

	std::cout << "solver is initialized" << std::endl;
	std::cout << "results will be saved in " << save_dir_.relative_path().c_str() << std::endl;

	{
	    std::ofstream ofs((save_dir_ / "state_space.dat").c_str());
	    boost::archive::text_oarchive oa(ofs);
	    oa << state.ptr_state_space_;
	}

	{
	    std::ofstream ofs((save_dir_ / "solver.dat").c_str());
	    boost::archive::text_oarchive oa(ofs);
	    oa << state.type;
	    oa << ode_system_.type;
	}
    }

    Logger::kLogger.add_stage("ODEIntegOther");
}


/**
 * TODO Specify save mode.
 */
template<class StateType, class SystemType>
typename Solver<StateType, SystemType>::ObservationPointList Solver<StateType, SystemType>::get_obspoints_constant_interval(const double t_end, const double dt, const unsigned int steps_per_save) const
{
    ObservationPointList obs_points;

    double t = ode_system_.get_state().get_t();

    const unsigned int max_steps_not_saved = std::max(steps_per_save - 1, static_cast<unsigned int>(1));

    unsigned int steps_not_saved = max_steps_not_saved;

    while (t <= t_end) {
	const bool is_saving_at_this_point = steps_not_saved >= max_steps_not_saved || t == t_end;

	const int save_mode = is_saving_at_this_point ? 0 : -1;

	obs_points.push_back_point(t, save_mode);

	if (is_saving_at_this_point) {
	    steps_not_saved = 0;
	} else {
	    ++steps_not_saved;
	}

	const double next_t = t + dt;

	if (t < t_end && t_end <= next_t) {
	    obs_points.push_back_point(t_end, 0);
	    break;
	} else {
	    t = next_t;
	}
    }

    return obs_points;
}


template<class StateType, class SystemType>
auto Solver<StateType, SystemType>::get_constant_stepper() const
{
    boost::numeric::odeint::runge_kutta4<typename SystemType::State, double, typename SystemType::State, double, typename SystemType::algebra_type> stepper;
    return SolverStepperWrappeFix(stepper);
}


template<class StateType, class SystemType>
auto Solver<StateType, SystemType>::get_adaptive_stepper(const double eps_rel, double eps_abs) const
{
    eps_abs = get_default_eps_abs(eps_rel, eps_abs);

    return SolverStepperWrapperAdaptive(
	    ode_integrator_function::make_controlled_rk_spmat<boost::numeric::odeint::runge_kutta_cash_karp54<typename SystemType::State, double, typename SystemType::State, double, typename SystemType::algebra_type>>(eps_abs, eps_rel),
	    eps_rel,
	    eps_abs
	);
}


template<class StateType, class SystemType>
template<class StepperType>
size_t Solver<StateType, SystemType>::solve(ObservationPointList& obs_points, StepperType& stepper)
{
    std::filesystem::path dir_result_saved = save_dir_ / "solved/";

    StateType& state = ode_system_.get_density_matrix();

    typename SystemType::State ODE_state(state);

    Observer observer(
	    ode_system_,
	    obs_points,
	    dir_result_saved,
	    force_saving_interval_
	);
    obs_points.set_start_point(state.get_t());


    if (mpi_helper::is_manager_rank()) {
	std::filesystem::create_directories(dir_result_saved / "tmp");

	{
	    std::ofstream ofs(dir_result_saved / "stepper.dat");
	    ofs << stepper.dump() << std::endl;
	}

	{
	    std::ofstream ofs(dir_result_saved / "observation_points.dat");
	    boost::archive::text_oarchive oa(ofs);
	    oa << obs_points;
	}
    }

    Logger::kLogger.push("ODEIntegOther");
    size_t n = ode_integrator_function::integrate_times(
	    boost::ref(stepper.stepper),
	    boost::ref(ode_system_),
	    ODE_state,
	    obs_points.begin(),
	    obs_points.end(),
	    obs_points.get_step(0) / 2,
	    boost::ref(observer)
	    );
    Logger::kLogger.pop();

    return n;
}


template<class StateType, class SystemType>
double Solver<StateType, SystemType>::get_default_eps_abs(const double eps_rel, const double eps_abs) const
{
    if (eps_abs >= eps_rel) {
	return 1.0 / ode_system_.get_density_matrix().get_n_dim_rho() * eps_rel;
    } else {
	return eps_abs;
    }
}


} // namespace cppoqss


#endif
