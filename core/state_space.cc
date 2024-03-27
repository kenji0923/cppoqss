#include <cppoqss/state_space.h>

#include <fstream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <cppoqss/arithmetic.h>
#include <cppoqss/mpi_helper.h>
#include <cppoqss/progress_bar.h>


namespace cppoqss {


StateSpaceRuntimeIndexGetter::StateSpaceRuntimeIndexGetter(const std::vector<std::unique_ptr<const ISingleStateSpace>>& state_spaces)
{
    for (int i = state_spaces.size() - 1; i >= 0; --i) {
	const auto& state_space = state_spaces[i];

	const MyIndexType n_dim = state_space->get_n_dim();
	const std::string space_name = state_space->get_space_name();

	map_spacename_to_n_dim_.emplace(space_name, n_dim);

	MyIndexType n_dim_product = 1;

	for (int j = state_spaces.size() - 1; j > i; --j) {
	    n_dim_product *= state_spaces[j]->get_n_dim();
	}

	map_spacename_to_divider_.emplace(space_name, n_dim_product);
    }
}


MyIndexType StateSpaceRuntimeIndexGetter::get_single_space_index(const MyIndexType sindex, const std::string& space_name) const
{
    return (sindex / map_spacename_to_divider_.at(space_name)) % map_spacename_to_n_dim_.at(space_name);
}


StateSpaceCachedIndexGetter::StateSpaceCachedIndexGetter(const std::vector<std::unique_ptr<const ISingleStateSpace>>& state_spaces)
{
    size_t n_dim = 0;

    for (const auto& space : state_spaces) {
	if (n_dim == 0) {
	    n_dim = space->get_n_dim();
	} else {
	    n_dim *= space->get_n_dim();
	}
    }


    for (const auto& state_space : state_spaces) {
	map_spacename_to_sindex_eigenvalueindex_.emplace(state_space->get_space_name(), std::vector<MyIndexType>(n_dim));
    }


    ProgressBar::InitBar("cache eigen value indices", n_dim);
    for (MyIndexType sindex = 0; sindex < n_dim; ++sindex) {
	MyIndexType sindex_copy = sindex;

	for (int i_space = state_spaces.size() - 1; i_space >= 0; --i_space) {
	    const MyIndexType n_dim_single = state_spaces[i_space]->get_n_dim();

	    map_spacename_to_sindex_eigenvalueindex_.at(state_spaces[i_space]->get_space_name())[sindex] = sindex_copy % n_dim_single;

	    sindex_copy = sindex_copy / n_dim_single;
	}

	ProgressBar::ProgressStep();
    }
}


MyIndexType StateSpaceCachedIndexGetter::get_single_space_index(const MyIndexType sindex, const std::string& space_name) const
{
    return map_spacename_to_sindex_eigenvalueindex_.at(space_name).at(sindex);
}


StateSpace::StateSpace(
    std::vector<std::unique_ptr<const ISingleStateSpace>>&& state_spaces,
    std::vector<std::unique_ptr<const ICOperator>>&& c_operators,
    const std::shared_ptr<IOperator>& hamiltonian,
    IndexGetterType index_getter_type
  )
: state_spaces_(std::move(state_spaces)),
  c_operators_(std::move(c_operators)),
  hamiltonian_(hamiltonian),
  index_getter_type_(index_getter_type)
{
    printf("initialize state space\n");

    for (const auto& state_space : state_spaces_) {
	if (state_space->get_n_dim() == 0) {
	    fprintf(stderr, "state space %s has 0 dimension\n", state_space->get_space_name().c_str());
	    exit(1);
	}
    }

    if (index_getter_type_ == IndexGetterType::Runtime) {
	index_getter_.reset(new StateSpaceRuntimeIndexGetter(state_spaces_));
    } else {
	index_getter_.reset(new StateSpaceCachedIndexGetter(state_spaces_));
    }

    initialize();
}


std::vector<std::reference_wrapper<const ICOperator>> StateSpace::get_c_operators() const
{
  std::vector<std::reference_wrapper<const ICOperator>> c_operators_ref;
  for (const auto& c_op : c_operators_) {
    c_operators_ref.push_back(std::cref(*c_op));
  }
  return c_operators_ref;
}


const std::string& StateSpace::get_outer_state_name(const MyIndexType index) const
{
  return outer_state_names_.at(index);
}


template<class Archive>
void save_construct_data(Archive& ar, const StateSpace* t, const unsigned int version)
{
    // ar << t->state_spaces_;
    // ar << t->c_operators_;
    ar << t->hamiltonian_;
    ar << t->index_getter_type_;
}


template<class Archive>
void load_construct_data(Archive& ar, StateSpace* t, const unsigned int version)
{
    std::vector<std::unique_ptr<const cppoqss::ISingleStateSpace>> state_spaces;
    std::vector<std::unique_ptr<const cppoqss::ICOperator>> c_operators;
    std::shared_ptr<cppoqss::IOperator> hamiltonian;
    cppoqss::StateSpace::IndexGetterType index_getter_type;

    // ar >> state_spaces;
    // ar >> c_operators;
    ar >> hamiltonian;
    ar >> index_getter_type;

    ::new(t)StateSpace(std::move(state_spaces), std::move(c_operators), hamiltonian, index_getter_type);
}


template void save_construct_data(boost::archive::text_oarchive& ar, const StateSpace* t, const unsigned int version);
template void load_construct_data(boost::archive::text_iarchive& ar, StateSpace* t, const unsigned int version);


void StateSpace::initialize()
{
    n_space_ = state_spaces_.size();

    n_dim_rho_ = 0;
    for (const auto& space : state_spaces_) {
	if (n_dim_rho_ == 0) {
	    n_dim_rho_ = space->get_n_dim();
	} else {
	    n_dim_rho_ *= space->get_n_dim();
	}
    }

    for (const auto& state_space : state_spaces_) {
	map_name_to_space_.emplace(state_space->get_space_name(), *state_space);
    }

    for (const auto& c_op : c_operators_) {
	const std::unordered_set<std::string> names = c_op->get_outer_state_names(*this);
	for (const auto& name : names) {
	    outer_state_names_.emplace_back(name);
	}
    }

    n_dim_phi_ = outer_state_names_.size();

    if (mpi_helper::is_printing_rank()) {
	printf("state spaces:\n");
	for (const auto& state_space : state_spaces_) {
	    printf("  %s: %ld dimension\n", state_space->get_space_name().c_str(), state_space->get_n_dim());
	}

	printf("%zu c-operators are found\n", c_operators_.size());

	printf("dimension of rho: %ld × %ld\n", n_dim_rho_, n_dim_rho_);
	printf("dimension of phi: %ld\n", n_dim_phi_);
    }
}


} // namespace cppoqss


#include <boost/serialization/export.hpp>


BOOST_CLASS_EXPORT_IMPLEMENT(cppoqss::IOperator)
