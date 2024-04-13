add_library(StateSystemName INTERFACE include/cppoqss/state_system_name.h)
target_include_directories(StateSystemName INTERFACE include)


add_library(Unit INTERFACE include/cppoqss/unit.h)
target_include_directories(Unit INTERFACE include)


add_library(Logger logger.cc include/cppoqss/logger.h)
target_include_directories(Logger PUBLIC include)
target_link_libraries(Logger PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
)
target_link_libraries(Logger PRIVATE
)


add_library(MPIHelper mpi_helper.cc include/cppoqss/mpi_helper.h)
target_include_directories(MPIHelper PUBLIC include)
target_link_libraries(MPIHelper PUBLIC
    Boost::mpi
    MPI::MPI_CXX
)
target_link_libraries(MPIHelper PRIVATE
    PkgConfig::PETSC
)


add_library(ProgressBar progress_bar.cc include/cppoqss/progress_bar.h)
target_include_directories(ProgressBar PUBLIC include)
target_link_libraries(ProgressBar PUBLIC
    indicators::indicators
)
target_link_libraries(ProgressBar PRIVATE
    MPIHelper
)


add_library(Arithmetic arithmetic.cc include/cppoqss/arithmetic.h)
target_include_directories(Arithmetic  PUBLIC include)
target_link_libraries(Arithmetic  PUBLIC
    PkgConfig::PETSC

    MPIHelper
)


add_library(StateSpace state_space.cc include/cppoqss/state_space.h)
target_include_directories(StateSpace PUBLIC include)
target_link_libraries(StateSpace PUBLIC
    cereal::cereal

    Arithmetic 
)
target_link_libraries(StateSpace PRIVATE
    MPIHelper
    ProgressBar
)


add_library(DensityMatrix density_matrix.cc include/cppoqss/density_matrix.h)
target_include_directories(DensityMatrix PUBLIC include)
target_link_libraries(DensityMatrix PUBLIC
    Arithmetic 
    MPIHelper
    StateSpace
)
target_link_libraries(DensityMatrix PRIVATE
    ProgressBar
    StateSystemName
)


add_library(Operator operator.cc include/cppoqss/operator.h)
target_include_directories(Operator PUBLIC include)
target_link_libraries(Operator PUBLIC
    cereal::cereal

    Arithmetic
    StateSpace
)


add_library(COperator INTERFACE include/cppoqss/c_operator.h)
target_include_directories(COperator INTERFACE include)
target_link_libraries(COperator INTERFACE
    Boost::boost

    Arithmetic
    MPIHelper
    ProgressBar
    StateSpace
)


add_library(Result result.cc include/cppoqss/result.h)
target_include_directories(Result PUBLIC include)
target_link_libraries(Result PUBLIC

    Arithmetic
    MPIHelper
)


add_library(OdeSystem ode_system.cc include/cppoqss/ode_system.h)
target_include_directories(OdeSystem PUBLIC include)
target_link_libraries(OdeSystem PUBLIC
    Arithmetic
    COperator
    DensityMatrix
    Operator
    Result
)
target_link_libraries(OdeSystem PRIVATE
    Logger
    StateSpace
    StateSystemName
)


add_library(OdeIntegratorFunction INTERFACE include/cppoqss/ode_integrator_function.h)
target_include_directories(OdeIntegratorFunction INTERFACE include)
target_link_libraries(OdeIntegratorFunction INTERFACE
    Boost::boost
)


add_library(Solver INTERFACE include/cppoqss/solver.h)
target_include_directories(Solver INTERFACE include)
target_link_libraries(Solver INTERFACE
    cereal::cereal
    Boost::boost

    COperator
    DensityMatrix
    Logger
    OdeIntegratorFunction
    Operator
    ProgressBar
    Result
    Unit
)


add_library(ResultAnalyzer result_analyzer.cc include/cppoqss/result_analyzer.h)
target_include_directories(ResultAnalyzer PUBLIC include)
target_link_libraries(ResultAnalyzer PUBLIC
    cereal::cereal

    Arithmetic
    OdeSystem
    StateSpace
    StateSystemName
)


if (${ROOT_FOUND})
    add_library(ResultAnalyzerUsingROOT result_analyzer_using_ROOT.cc include/cppoqss/result_analyzer_using_ROOT.h)
    target_include_directories(ResultAnalyzerUsingROOT PUBLIC include)
    target_link_libraries(ResultAnalyzerUsingROOT PUBLIC
	ROOT::Hist

	ResultAnalyzer
    )
endif()


add_library(Simulator INTERFACE)
target_link_libraries(Simulator INTERFACE Solver OdeSystem COperator Operator DensityMatrix StateSpace)

add_library(Analyzer INTERFACE)
target_link_libraries(Analyzer INTERFACE ResultAnalyzer)
if (${ROOT_FOUND})
    target_link_libraries(Analyzer INTERFACE ResultAnalyzerUsingROOT)
endif()


add_library(cppoqss::Simulator ALIAS Simulator)

add_library(cppoqss::Analyzer ALIAS Analyzer)
