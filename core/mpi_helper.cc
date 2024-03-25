#include <cppoqss/mpi_helper.h>

#include <chrono>

#include <petscsys.h>


namespace cppoqss {


IsRequireMPI::IsRequireMPI()
{
  objects_require_MPI.emplace(this);
}


IsRequireMPI::~IsRequireMPI()
{
  objects_require_MPI.erase(this);
  if (mpi_helper::is_finalized() && get_n_obj() == 0) PetscFinalize();
}


std::unordered_set<IsRequireMPI*> IsRequireMPI::objects_require_MPI;


void mpi_helper::initialize(int* argc, char*** args, const char file[], const char help[])
{
  // TODO Switch initializer.
  PetscInitialize(argc, args, file, help);
  initialized_time = std::chrono::high_resolution_clock::now();
}


void mpi_helper::finalize()
{
  is_ready_for_finalizing = true;
  if (IsRequireMPI::get_n_obj() == 0) PetscFinalize();
}


void mpi_helper::print_elapsed_time(const std::string& message)
{
  if (is_manager_rank()) {
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto timer_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - initialized_time);
    if (message == "") {
      printf("%.1lfs elapsed\n", (double)timer_elapsed.count() / 1000);
    } else {
      printf("%s at %.1lfs elapsed\n", message.c_str(), (double)timer_elapsed.count() / 1000);
    }
  }
}


boost::mpi::communicator mpi_helper::world;


bool mpi_helper::is_printing_to_only_manager = false;


size_t mpi_helper::manager_rank = 0;


std::chrono::high_resolution_clock::time_point mpi_helper::initialized_time;


bool mpi_helper::is_ready_for_finalizing = false;


} // namespace cppoqss
