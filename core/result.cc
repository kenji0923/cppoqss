#include <cppoqss/result.h>


#include <filesystem>
#include <fstream>
#include <memory>
#include <sys/types.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>

#include <cppoqss/mpi_helper.h>


namespace cppoqss {


// Result::Result(const Info& given_info)
// : time_point(time_point_name), rho_offset(rho_offset_name), phi_offset(phi_offset_name), time_steps(time_steps_name), failed_time_steps(failed_time_steps_name), info(info_name)
// {
//     info.data = given_info;
// }
// 
// 
// void Result::save(std::filesystem::path dir_saving_at) const
// {
//     time_point.save(dir_saving_at.remove_filename());
//     rho_offset.save(dir_saving_at.remove_filename());
//     phi_offset.save(dir_saving_at.remove_filename());
//     time_steps.save(dir_saving_at.remove_filename());
//     failed_time_steps.save(dir_saving_at.remove_filename());
//     info.save(dir_saving_at.remove_filename());
// }
// 
// 
// void Result::add_state(const double new_time_point, const off_t new_rho_offset, const off_t new_phi_offset)
// {
//     time_point.data.push_back(new_time_point);
//     rho_offset.data.push_back(new_rho_offset);
//     phi_offset.data.push_back(new_phi_offset);
// }
// 
// 
// void Result::add_time_step(const double t, const double dt)
// {
//     time_steps.data.push_back({ t, dt });
// }
// 
// 
// void Result::add_failed_time_step(const double t, const double dt_tried, const double dt_new)
// {
//     failed_time_steps.data.push_back({ t, dt_tried, dt_new });
// }
// 
// 
// size_t Result::get_index(const double t_query) const
// {
//     auto it_lower_bound = std::lower_bound(time_point.data.begin(), time_point.data.end(), t_query);
// 
//     if (it_lower_bound == time_point.data.begin()) return 0;
//     if (it_lower_bound == time_point.data.end()) return time_point.data.size() - 1;
// 
//     const double t_early = *(it_lower_bound - 1);
//     const double t_later = *(it_lower_bound);
// 
//     const size_t distance = std::distance(time_point.data.begin(), it_lower_bound);
// 
//     if (t_query - t_early < t_later - t_query) {
// 	return distance - 1;
//     } else {
// 	return distance;
//     }
// }
// 
// 
// LoadedResult::LoadedResult(const std::filesystem::path& solved_path)
// : solved_path_(solved_path)
// {
//     auto load_serialized_data = [this](auto& target, const std::filesystem::path& file_basename)
//     {
// 	std::ifstream ifs((this->solved_path_/ file_basename).c_str());
// 	{
// 	    boost::archive::text_iarchive ia(ifs);
// 	    ia >> target.data;
// 	}
//     };
// 
//     load_serialized_data(result_.time_point, Result::time_point_name);
//     load_serialized_data(result_.rho_offset, Result::rho_offset_name);  
//     load_serialized_data(result_.phi_offset, Result::phi_offset_name);  
//     load_serialized_data(result_.time_steps, Result::time_steps_name);  
//     load_serialized_data(result_.failed_time_steps, Result::failed_time_steps_name);
//     load_serialized_data(result_.info, Result::info_name);
// 
//     if (mpi_helper::is_printing_rank()) {
// 	printf("analyzer for %s initiallized\n", solved_path_.c_str());
// 	printf("%zu points found\n", result_.time_point.data.size());
//     }
// }
// 
// 
// std::unique_ptr<IResultState> LoadedResult::get_rho(const size_t index) const
// {
//     off_t offset = result_.rho_offset.data[index];
// 
//     if (result_.info.data.is_diagonal_saved) {
// 	return std::make_unique<ResultState<MyVec>>(solved_path_, offset);
//     } else {
// 	return std::make_unique<ResultState<MyMat>>(solved_path_, offset);
//     }
// }
// 
// 
// std::unique_ptr<MyVec> LoadedResult::get_phi(const size_t index) const
// {
//     off_t offset = result_.phi_offset.data[index];
// 
//     return std::make_unique<MyVec>(solved_path_ / Result::phi_name, offset);
// }


} // namespace cppoqss
