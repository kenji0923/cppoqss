#ifndef RESULT_H
#define RESULT_H


#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sys/types.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/utility.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/mpi_helper.h>


namespace cppoqss {


template<class DataType>
struct ResultData
{
  ResultData() { }
  ResultData(const std::filesystem::path& filename) : filename(filename) { }

  void save(const std::filesystem::path& dir_saving_at) const;

  DataType data;
  std::filesystem::path filename;
};


template<class DataType>
void ResultData<DataType>::save(const std::filesystem::path& dir_saving_at) const
{
    std::filesystem::path save_filepath = dir_saving_at / filename;
    std::filesystem::path tmp_filepath = std::string(save_filepath.c_str()) + ".tmp";

    if (mpi_helper::is_manager_rank()) {
	std::ofstream ofs(tmp_filepath.c_str());
	boost::archive::text_oarchive oa(ofs);
	oa << data;
	std::filesystem::rename(tmp_filepath, save_filepath);
    }
}


template<class StateType>
struct ResultMeta
{
    ResultMeta(
	    const std::filesystem::path& default_parent_directory,
	    const std::string& save_prefix,
	    const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& append_saver,
	    const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& rewrite_saver
	);

    void append(const StateType& state, const int save_mode);
    void rewrite_data_mode0(const StateType& state, const std::filesystem::path& save_parent_directory);
    
    off_t call_saver(const StateType& state, const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& saver, const int save_mode, const std::filesystem::path& save_parent_directory);

    ResultData<std::vector<off_t>> offsets;
    ResultData<std::vector<int>> save_modes;

    std::filesystem::path default_parent_directory;
    std::string save_prefix;
    std::function<off_t(const StateType&, const int, const std::filesystem::path&)> append_saver;
    std::function<off_t(const StateType&, const int, const std::filesystem::path&)> rewrite_saver;
    std::string data_filename = "data.dat";
};


template<class StateType>
ResultMeta<StateType>::ResultMeta(
	const std::filesystem::path& default_parent_directory,
	const std::string& save_prefix,
	const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& append_saver,
	const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& rewrite_saver
	)
:   offsets(save_prefix + "_offsets.dat"),
    save_modes(save_prefix + "_save_modes.dat"),
    default_parent_directory(default_parent_directory),
    save_prefix(save_prefix),
    append_saver(append_saver),
    rewrite_saver(rewrite_saver)
{ }


template<class StateType>
void ResultMeta<StateType>::append(const StateType& state, const int save_mode)
{
    off_t offset = call_saver(state, append_saver, save_mode, default_parent_directory);

    offsets.data.push_back(offset);
    offsets.save(default_parent_directory);

    save_modes.data.push_back(save_mode);
    save_modes.save(default_parent_directory);
}


template<class StateType>
void ResultMeta<StateType>::rewrite_data_mode0(const StateType& state, const std::filesystem::path& save_parent_directory)
{
    call_saver(state, rewrite_saver, 0, save_parent_directory);
}


template<class StateType>
off_t ResultMeta<StateType>::call_saver(const StateType& state, const std::function<off_t(const StateType&, const int, const std::filesystem::path&)>& saver, const int save_mode, const std::filesystem::path& save_parent_directory)
{
    const std::filesystem::path save_filepath = save_parent_directory / (save_prefix + "_" + data_filename);

    off_t offset = saver(state, save_mode, save_filepath);

    return offset;
}


// class IResultState
// {
// public:
//     virtual ~IResultState() { }
// 
//     virtual MyRow get_row(const MyIndexType i) const = 0;
//     virtual void restore_row(const MyIndexType i, MyRow& row) const = 0;
// };


// class LoadedResult
// {
// public:
//     LoadedResult(const std::filesystem::path& solved_path);
// 
//     const Result& get_result() const { return result_; }
// 
//     size_t get_index(const double t_query) const { return result_.get_index(t_query); }
//     std::unique_ptr<IResultState> get_rho(const size_t index) const;
//     std::unique_ptr<MyVec> get_phi(const size_t index) const;
// 
// private:
//     std::filesystem::path solved_path_;
//     Result result_;
// };


// template<class StateType>
// class ResultState: public IResultState
// {
// public:
//   ResultState(const std::filesystem::path& solved_path, const off_t offset);
//   MyRow get_row(const MyIndexType i) const override;
//   void restore_row(const MyIndexType i, MyRow& row) const override;
// 
// private:
//   StateType rho_;
// };
// 
// 
// template<class RhoType>
// ResultState<RhoType>::ResultState(const std::filesystem::path& solved_path, const off_t offset)
// : rho_(solved_path / Result::rho_name, offset)
// { }
// 
// 
// template<class RhoType>
// MyRow ResultState<RhoType>::get_row(const MyIndexType i) const
// {
//   return rho_.get_row(i);
// }
// 
// 
// template<class RhoType>
// void ResultState<RhoType>::restore_row(const MyIndexType i, MyRow& row) const
// {
//   rho_.restore_row(i, row);
// }
// 
// 
// template<>
// MyRow ResultState<MyVec>::get_row(const MyIndexType i) const
// {
//   return rho_.get_row();
// }
// 
// 
// template<>
// void ResultState<MyVec>::restore_row(const MyIndexType i, MyRow& row) const
// {
//   rho_.restore_row(row);
// }


} // namespace cppoqss


#endif
