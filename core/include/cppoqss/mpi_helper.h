#ifndef MPI_HELPER_H
#define MPI_HELPER_H


#include <chrono>
#include <string>
#include <unordered_set>

#include <boost/mpi.hpp>


namespace cppoqss {


class IsRequireMPI
{
public:
  static size_t get_n_obj() { return objects_require_MPI.size(); }

  IsRequireMPI();
  virtual ~IsRequireMPI();

private:
  static std::unordered_set<IsRequireMPI*> objects_require_MPI;
};

class mpi_helper
{
public:
  static void initialize(int* argc, char*** args, const char file[], const char help[]);
  static void finalize();

  static size_t get_rank() { return world.rank(); }
  static size_t get_manager_rank() { return manager_rank; }
  static size_t get_size() { return world.size(); }

  static bool is_manager_rank() { return get_rank() == manager_rank; }
  static bool is_printing_rank() { return is_printing_to_only_manager ? is_manager_rank() : true; }
  static bool is_finalized() { return is_ready_for_finalizing; }

  static void print_elapsed_time(const std::string& message);

  static boost::mpi::communicator world;
  static bool is_printing_to_only_manager;

private:
  static size_t manager_rank;
  static std::chrono::high_resolution_clock::time_point initialized_time;
  static bool is_ready_for_finalizing;
};


} // namespace cppoqss


#endif
