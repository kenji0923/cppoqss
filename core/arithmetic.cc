#include <cppoqss/arithmetic.h>

#include <cassert>
#include <functional>
#include <tuple>
#include <vector>

#include <boost/mpi.hpp>
#include <petscmat.h>
#include <petscvec.h>
#include <petscsystypes.h>

#include <cppoqss/mpi_helper.h>


namespace cppoqss {


#ifdef USE_PETSC


MyRowCopied MyRow::copy() const
{
  MyRowCopied copied;
  copied.n_col = n_col;
  copied.cols.resize(n_col);
  copied.values.resize(n_col);
  for (MyIndexType ii = 0; ii < n_col; ++ii) {
    copied.cols[ii] = cols_ptr[ii];
    copied.values[ii] = values_ptr[ii];
  }
  return copied;
}


MyVec::Ownership::Ownership(const Vec& vec)
{
  if (vec) {
    MyIndexType m;
    VecGetSize(vec, &m);

    MyIndexType n_ownership = PETSC_DECIDE;
    PetscSplitOwnership(PETSC_COMM_WORLD, &n_ownership, &m);

    MyIndexType start = boost::mpi::scan(mpi_helper::world, n_ownership, std::plus<MyIndexType>()) - n_ownership;
    MyIndexType end = start + n_ownership;

    n_dim = m;
    n = end - start;
    range[0] = start;
    range[1] = end;
    for (MyIndexType i = start; i < end; ++i) {
      indices.emplace_back(i);
    }
  } else {
    n_dim = 0;
    n = 0; 
    range[0] = 0;
    range[1] = 0;
  }
}

MyVec::MyVec(const MyIndexType n_dim)
: vec_(nullptr), vec_owned_(nullptr)
{
  VecCreate(PETSC_COMM_WORLD, &vec_owned_);
  VecSetSizes(vec_owned_, PETSC_DECIDE, n_dim);
  VecSetFromOptions(vec_owned_);

  initialize_by_owned();
}

MyVec::MyVec(const Vec vec)
: vec_(vec), vec_owned_(nullptr), ownership_info_(vec)
{ }

MyVec::MyVec(const std::filesystem::path& read_path, const off_t offset)
{
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, read_path.c_str(), FILE_MODE_READ, &viewer);

  int fd;
  PetscViewerBinaryGetDescriptor(viewer, &fd);

  off_t new_offset;
  PetscBinarySeek(fd, offset, PETSC_BINARY_SEEK_SET, &new_offset);

  VecCreate(PETSC_COMM_WORLD, &vec_owned_);
  VecLoad(vec_owned_, viewer);

  initialize_by_owned();

  PetscViewerDestroy(&viewer);
}

MyVec::MyVec(const MyVec& rh)
: vec_(nullptr), vec_owned_(nullptr), ownership_info_()
{
  *this = rh;
}

MyVec::MyVec(MyVec&& rh)
: vec_(nullptr), vec_owned_(nullptr), ownership_info_()
{
  *this = std::move(rh);
}

MyVec::~MyVec()
{
  if (vec_owned_) {
    VecDestroy(&vec_owned_);
  }
}

MyVec& MyVec::operator=(const MyVec& rh)
{
    *this = rh.duplicate();
    return *this;
}

MyVec& MyVec::operator=(MyVec&& rh)
{
    if (vec_owned_) {
	VecDestroy(&vec_owned_);
    }

    vec_ = rh.vec_;
    vec_owned_ = rh.vec_owned_;
    ownership_info_ = rh.ownership_info_;

    rh.vec_ = nullptr;
    rh.vec_owned_ = nullptr;

    return *this;
}

MyVec MyVec::duplicate() const
{
  MyVec dup_vec(nullptr);

  VecDuplicate(vec_, &dup_vec.vec_owned_);
  VecCopy(vec_, dup_vec.vec_owned_);

  dup_vec.initialize_by_owned();

  return dup_vec;
}

MyVec& MyVec::add_ax(const MyElementType a, const MyVec& x)
{
  VecAXPY(vec_, a, x.vec_);
  return *this;
}

MyVec& MyVec::add_to_all_elements(const MyElementType add_value)
{
  VecShift(vec_, add_value);
  return *this;
}

MyVec& MyVec::pointwise_divide(const MyVec& denominator) 
{
  VecPointwiseDivide(this->vec_, this->vec_, denominator.vec_);
  return *this;
}

MyVec& MyVec::scale(const MyElementType scale_factor)
{
  VecScale(vec_, scale_factor);
  return *this;
}

MyVec& MyVec::convert_to_abs()
{
  VecAbs(vec_);
  return *this;
}

MyRow MyVec::get_row() const
{
  MyRow row { ownership_info_.n, &ownership_info_.indices[0], nullptr };
  VecGetArrayRead(vec_, &row.values_ptr);
  return row;
}

void MyVec::restore_row(MyRow& row) const
{
  VecRestoreArrayRead(vec_, &row.values_ptr);
}

void MyVec::loop_over_elements(std::function<void(const MyIndexType, const MyElementType)> process) const
{
  MyRow elements = get_row();
  for (MyIndexType ii = 0; ii < ownership_info_.n; ++ii) {
    process(elements.cols_ptr[ii], elements.values_ptr[ii]);
  }
  restore_row(elements);
}

void MyVec::set_values(const MyIndexType n_element, const MyIndexType* indices_ptr, const MyElementType* values_ptr)
{
  VecSetValues(vec_, n_element, indices_ptr, values_ptr, INSERT_VALUES);
}

void MyVec::add_values(const MyIndexType n_element, const MyIndexType* indices_ptr, const MyElementType* values_ptr)
{
  VecSetValues(vec_, n_element, indices_ptr, values_ptr, ADD_VALUES);
}

void MyVec::write_to_file(const std::filesystem::path& path) const
{
  std::filesystem::path rho_path_tmp = std::string(path.c_str()) + ".tmp";
  std::filesystem::path info_path_tmp = std::string(rho_path_tmp.c_str()) + ".info";
  std::filesystem::path info_path = std::string(path.c_str()) + ".info";

  PetscViewer viewer = NULL;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, rho_path_tmp.c_str(), FILE_MODE_WRITE, &viewer);

  VecView(vec_, viewer);

  PetscViewerDestroy(&viewer);

  if (mpi_helper::is_manager_rank()) {
    std::filesystem::rename(rho_path_tmp, path);
    std::filesystem::rename(info_path_tmp, info_path);
  }
}

off_t MyVec::append_to_file(const std::filesystem::path& path) const
{
  PetscViewer viewer = NULL;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, path.c_str(), FILE_MODE_APPEND, &viewer);

  int fd;
  off_t offset;

  PetscViewerBinaryGetDescriptor(viewer, &fd);
  PetscBinarySeek(fd, 0, PETSC_BINARY_SEEK_END, &offset);

  VecView(vec_, viewer);

  PetscViewerDestroy(&viewer);

  return offset;
}

void MyVec::initialize_by_owned()
{
  vec_ = vec_owned_;
  ownership_info_ = Ownership(vec_);
}

MyMat::Ownership::Ownership(const Mat& mat)
{
  if (mat) {
    MyIndexType m, n;
    MatGetSize(mat, &m, &n);
    assert(m == n);

    MyIndexType n_ownership_row = PETSC_DECIDE;
    PetscSplitOwnership(PETSC_COMM_WORLD, &n_ownership_row, &m);

    MyIndexType row_start = boost::mpi::scan(mpi_helper::world, n_ownership_row, std::plus<MyIndexType>()) - n_ownership_row;
    MyIndexType row_end = row_start + n_ownership_row;

    n_row = row_end - row_start;
    range_row[0] = row_start;
    range_row[1] = row_end;

    n_dim = m;
    n_col = n;
    range_col[0] = 0;
    range_col[1] = n;
  } else {
    n_dim = 0;
    n_col = 0;
    range_col[0] = 0;
    range_col[1] = 0;
  }
}

std::tuple<MyIndexType, MyIndexType> MyMat::get_ownership_range_before_preallocation(const MyMat& mat)
{
  MyIndexType m, n;
  MatGetSize(mat.mat_, &m, &n);
  assert(m == n);

  MyIndexType n_ownership_row = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD, &n_ownership_row, &m);

  MyIndexType row_start = boost::mpi::scan(mpi_helper::world, n_ownership_row, std::plus<MyIndexType>()) - n_ownership_row;
  MyIndexType row_end = row_start + n_ownership_row;

  return {row_start, row_end};
}

MyMat MyMat::MatMatMulti(const MyMat& mat_left, const MyMat& mat_right)
{
  MyMat mat_result(nullptr);

  MatMatMult(mat_left.mat_, mat_right.mat_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mat_result.mat_owned_);
  mat_result.initialize_by_owned();

  return mat_result;
}

MyMat MyMat::HermitianTranspose(const MyMat& mat_source)
{
  MyMat mat_result(nullptr);

  MatHermitianTranspose(mat_source.mat_, MAT_INITIAL_MATRIX, &mat_result.mat_owned_);
  mat_result.initialize_by_owned();

  return mat_result;
}
  
MatType MyMat::DefaultMatType = 
    // MATAIJKOKKOS
    MATMPIAIJ
  ;


MyMat::MyMat(const MyIndexType n_dim, bool is_hermite)
: mat_(nullptr), mat_owned_(nullptr), is_hermite_(is_hermite)
{
    MatCreate(PETSC_COMM_WORLD, &mat_owned_);
    MatSetSizes(mat_owned_, PETSC_DECIDE, PETSC_DECIDE, n_dim, n_dim);
    MatSetType(mat_owned_, DefaultMatType);
    if (is_hermite) MatSetOption(mat_owned_, MAT_HERMITIAN, PETSC_TRUE);

    initialize_by_owned();
}


MyMat::MyMat(const Mat mat)
: mat_(mat), mat_owned_(nullptr), ownership_info_(mat), is_hermite_(false)
{
}


MyMat::MyMat(const std::filesystem::path& read_path, const off_t offset)
: is_hermite_(false)
{
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, read_path.c_str(), FILE_MODE_READ, &viewer);

    int fd;
    PetscViewerBinaryGetDescriptor(viewer, &fd);

    off_t new_offset;
    PetscBinarySeek(fd, offset, PETSC_BINARY_SEEK_SET, &new_offset);

    MatCreate(PETSC_COMM_WORLD, &mat_owned_);
    MatLoad(mat_owned_, viewer);

    initialize_by_owned();

    PetscViewerDestroy(&viewer);
}


MyMat::MyMat(const MyMat& rh)
: mat_(nullptr), mat_owned_(nullptr), ownership_info_(), is_hermite_(false)
{
    *this = rh;
}


MyMat::MyMat(MyMat&& rh)
: mat_(nullptr), mat_owned_(nullptr), ownership_info_(), is_hermite_(false)
{
    *this = std::move(rh);
}


MyMat::~MyMat()
{
    if (mat_owned_) {
	MatDestroy(&mat_owned_);
    }
}


MyMat& MyMat::operator=(const MyMat& rh)
{
    *this = rh.duplicate();
    return *this;
}


MyMat& MyMat::operator=(MyMat&& rh)
{
    if (mat_owned_) {
	MatDestroy(&mat_owned_);
    }

    mat_ = rh.mat_;
    mat_owned_ = rh.mat_owned_;
    ownership_info_ = Ownership(mat_);

    rh.mat_ = nullptr;
    rh.mat_owned_ = nullptr;

    return *this;
}


void MyMat::reset()
{
    MatZeroEntries(mat_);
}


MyMat MyMat::duplicate() const
{
    MyMat new_mat(nullptr);

    MatConvert(mat_, MATSAME, MAT_INITIAL_MATRIX, &new_mat.mat_owned_);

    new_mat.initialize_by_owned();

    return new_mat;
}


MyVec MyMat::get_diagonal() const
{
    Vec vec;
    VecCreate(PETSC_COMM_WORLD, &vec);
    VecSetSizes(vec, PETSC_DECIDE, get_n_dim());
    VecSetFromOptions(vec);

    MatGetDiagonal(mat_, vec);

    MyVec my_vec(vec);

    return my_vec;
}


MyVec MyMat::get_diagonal_to_local_vector() const
{
    Vec vec;
    VecCreate(PETSC_COMM_WORLD, &vec);
    VecSetSizes(vec, PETSC_DECIDE, get_n_dim());
    VecSetFromOptions(vec);

    MatGetDiagonal(mat_, vec);

    Vec v = vec;

    IS is;
    ISCreateStride(PETSC_COMM_WORLD, get_n_dim(), 0, 1, &is);

    Vec v_gathered;
    VecCreate(PETSC_COMM_SELF, &v_gathered);
    VecSetSizes(v_gathered, PETSC_DECIDE, get_n_dim());

    PetscBool is_kokkos_type_is_matched;
    PetscObjectBaseTypeCompareAny((PetscObject)v, &is_kokkos_type_is_matched, VECMPIKOKKOS, VECSEQKOKKOS, VECKOKKOS, "");
    if (is_kokkos_type_is_matched) {
	VecSetType(v_gathered, VECSEQKOKKOS);
    }
    if (!is_kokkos_type_is_matched) {
	VecSetType(v_gathered, VECSEQ);
    }

    VecScatter ctx;
    VecScatterCreate(v, is, v_gathered, is, &ctx);
    VecScatterBegin(ctx, v, v_gathered, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, v, v_gathered, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterDestroy(&ctx);

    ISDestroy(&is);

    MyVec my_vec(v_gathered);

    return my_vec;
}


MyIndexType MyMat::get_number_of_nonzeros() const
{
    MyIndexType number_of_nonzeros = 0;

    loop_over_nonzero_elements(
	    [&number_of_nonzeros](const MyIndexType i, const MyIndexType j, const MyElementType)
	    {
		++number_of_nonzeros;
	    }
	);

    MyIndexType number_of_nonzeros_sum;
    boost::mpi::all_reduce(mpi_helper::world, number_of_nonzeros, number_of_nonzeros_sum, std::plus<MyIndexType>());

    return number_of_nonzeros_sum;
}


MyMat::NonzeroNumbers MyMat::get_number_of_nonzeros_structured() const
{
    NonzeroNumbers nonzero_numbers(ownership_info_.n_row);

    for (MyIndexType i = ownership_info_.range_row[0]; i < ownership_info_.range_row[1]; ++i) {
	MyRow row = get_row(i);

	for (MyIndexType jj = 0; jj < row.n_col; ++jj) {
	    const MyIndexType j = row.cols_ptr[jj];

	    if (ownership_info_.range_row[0] <= j && j < ownership_info_.range_row[1]) {
		++nonzero_numbers.diag[i - ownership_info_.range_row[0]];
	    } else {
		++nonzero_numbers.nondiag[i - ownership_info_.range_row[0]];
	    }
	}

	restore_row(i, row);
    }

    return nonzero_numbers;
}


MyRow MyMat::get_row(const MyIndexType i) const
{
  MyRow row;
  if (is_hermite_) {
    MatGetRowUpperTriangular(mat_);
  }
  MatGetRow(mat_, i, &row.n_col, &row.cols_ptr, &row.values_ptr);

  return row;
  // std::tuple<std::vector<MyIndexType>, std::vector<MyElementType>> row_data;
  // std::get<0>(row_data).reserve(ncols);
  // std::get<1>(row_data).reserve(ncols);
  // for (MyIndexType jj = 0; jj < ncols; ++jj) {
  //   std::get<0>(row_data).push_back(cols[jj]);
  //   std::get<1>(row_data).push_back(vals[jj]);
  // }
}

void MyMat::restore_row(const MyIndexType i, MyRow& row) const
{
  MatRestoreRow(mat_, i, &row.n_col, &row.cols_ptr, &row.values_ptr);
  if (is_hermite_) {
    MatRestoreRowUpperTriangular(mat_);
  }
}

void MyMat::set_values(const MyIndexType n_row, const MyIndexType* ptr_rows, const MyIndexType n_col, const MyIndexType* ptr_cols, const MyElementType* ptr_elements)
{
  MatSetValues(mat_, n_row, ptr_rows, n_col, ptr_cols, ptr_elements, INSERT_VALUES);
}

void MyMat::add_values(const MyIndexType n_row, const MyIndexType* ptr_rows, const MyIndexType n_col, const MyIndexType* ptr_cols, const MyElementType* ptr_elements)
{
  MatSetValues(mat_, n_row, ptr_rows, n_col, ptr_cols, ptr_elements, ADD_VALUES);
}

MyMat& MyMat::add_ax(const MyElementType a, const MyMat& x)
{
  MatAXPY(mat_, a, x.mat_, SUBSET_NONZERO_PATTERN);
  return *this;
}

MyMat& MyMat::add_to_all_elements(const MyElementType add_value)
{
  // TODO Update to cppoqss.
  if (add_value != MyElementType(0.0)) {
    PetscInt i, start, end, j, ncols, m, n;
    const PetscInt *row;
    PetscScalar *val;
    const PetscScalar *vals;

    Mat& X = mat_;
    MatGetSize(X, &m, &n);
    MatGetOwnershipRange(X, &start, &end);

    MatGetRowUpperTriangular(X);
    for (i = start; i < end; i++) {
      MatGetRow(X, i, &ncols, &row, &vals);
      MatSetValues(X, 1, &i, ncols, row, &add_value, ADD_VALUES);
      MatRestoreRow(X, i, &ncols, &row, &vals);
    }
    MatRestoreRowUpperTriangular(X);

    MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);
  }
  return *this;
}

MyMat& MyMat::scale(const MyElementType scale_factor)
{
    MatScale(mat_, scale_factor);
    return *this;
}

void MyMat::set_preallocation(const NonzeroNumbers& nonzero_numbers)
{
  set_preallocation(0, &nonzero_numbers.diag[0], 0, &nonzero_numbers.nondiag[0]);
}

void MyMat::set_preallocation(const MyIndexType n_diag_nonzero, const MyIndexType* array_diag_nonzero, const MyIndexType n_nondiag_nonzero, const MyIndexType* array_nondiag_nonzero)
{
  MatMPIAIJSetPreallocation(mat_, n_diag_nonzero, array_diag_nonzero, n_nondiag_nonzero, array_nondiag_nonzero);
}

void MyMat::loop_over_elements(std::function<void(const MyIndexType i, const MyIndexType j)> process)
{
  for (MyIndexType i = ownership_info_.range_row[0]; i < ownership_info_.range_row[1]; ++i) {
    for (MyIndexType j = ownership_info_.range_col[0]; j < ownership_info_.range_col[1]; ++j) {
      process(i, j);
    }
  }
}

void MyMat::loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*, const MyElementType*)> process) const
{
  if (is_hermite_) MatGetRowUpperTriangular(mat_);
  for (MyIndexType i = ownership_info_.range_row[0]; i < ownership_info_.range_row[1]; ++i) {
    MyIndexType ncols;
    const MyIndexType *cols;
    const MyElementType *vals;

    MatGetRow(mat_, i, &ncols, &cols, &vals);
    process(i, ncols, cols, vals);
    MatRestoreRow(mat_, i, &ncols, &cols, &vals);
  }
  if (is_hermite_) MatRestoreRowUpperTriangular(mat_);
}

void MyMat::loop_over_nonzero_elements(std::function<void(const MyIndexType i, const MyIndexType j, const MyElementType element)> process) const
{
  if (is_hermite_) MatGetRowUpperTriangular(mat_);
  for (MyIndexType i = ownership_info_.range_row[0]; i < ownership_info_.range_row[1]; ++i) {
    MyIndexType ncols;
    const MyIndexType *cols;
    const MyElementType *vals;

    MatGetRow(mat_, i, &ncols, &cols, &vals);
    for (MyIndexType jj = 0; jj < ncols; ++jj) {
      process(i, cols[jj], vals[jj]);
    }
    MatRestoreRow(mat_, i, &ncols, &cols, &vals);
  }
  if (is_hermite_) MatRestoreRowUpperTriangular(mat_);
}

void MyMat::modify_nonzero_elements(std::function<MyElementType(const MyIndexType, const MyIndexType, const MyElementType)> modifier)
{
  SparseMatrix<MyElementType> new_matrix;

  if (is_hermite_) MatGetRowUpperTriangular(mat_);
  for (MyIndexType i = ownership_info_.range_row[0]; i < ownership_info_.range_row[1]; ++i) {
    MyIndexType ncols;
    const MyIndexType *cols;
    const MyElementType *vals;

    MatGetRow(mat_, i, &ncols, &cols, &vals);
    std::vector<MyElementType> new_vals(ncols);
    for (MyIndexType jj = 0; jj < ncols; ++jj) {
      new_matrix.push_back_element(cols[jj], modifier(i, cols[jj], vals[jj]));
    }
    MatRestoreRow(mat_, i, &ncols, &cols, &vals);

    new_matrix.push_back_row();
  }
  if (is_hermite_) MatRestoreRowUpperTriangular(mat_);

  new_matrix.loop_by_row(
      [this](const MyIndexType ii, const MyIndexType n_element, const MyIndexType* cols, const MyElementType* elements) {
        const MyIndexType i = ownership_info_.range_row[0] + ii;
        this->set_values(1, &i, n_element, cols, elements);
      }
    );

  begin_final_assembly();
  end_final_assembly();
}

MyMat::NonzeroNumbers MyMat::get_nonzero_numbers(std::function<bool(const MyIndexType, const MyIndexType)> checker_gives_true_if_nonzero)
{
  std::vector<MyIndexType> n_diag_nonzero(ownership_info_.n_row, 0);
  std::vector<MyIndexType> n_nondiag_nonzero(ownership_info_.n_row, 0);

  loop_over_elements(
      [&n_diag_nonzero, &n_nondiag_nonzero, &checker_gives_true_if_nonzero, this](const MyIndexType i, const MyIndexType j) {
        if (checker_gives_true_if_nonzero(i, j)) {
          const MyIndexType ii = i - this->ownership_info_.range_row[0];
          if (this->ownership_info_.range_row[0] <= j && j < this->ownership_info_.range_row[1]) {
            ++n_diag_nonzero[ii];
          } else {
            ++n_nondiag_nonzero[ii];
          }
        }
      }
    );

  return NonzeroNumbers(std::move(n_diag_nonzero), std::move(n_nondiag_nonzero));
}

void MyMat::write_to_file(const std::filesystem::path& path) const
{
  std::filesystem::path rho_path_tmp = std::string(path.c_str()) + ".tmp";
  std::filesystem::path info_path_tmp = std::string(rho_path_tmp.c_str()) + ".info";
  std::filesystem::path info_path = std::string(path.c_str()) + ".info";

  PetscViewer viewer = NULL;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, rho_path_tmp.c_str(), FILE_MODE_WRITE, &viewer);

  MatView(mat_, viewer);

  PetscViewerDestroy(&viewer);

  if (mpi_helper::is_manager_rank()) {
    std::filesystem::rename(rho_path_tmp, path);
    std::filesystem::rename(info_path_tmp, info_path);
  }
}

off_t MyMat::append_to_file(const std::filesystem::path& path) const
{
  PetscViewer viewer = NULL;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, path.c_str(), FILE_MODE_APPEND, &viewer);

  int fd;
  off_t offset;

  PetscViewerBinaryGetDescriptor(viewer, &fd);
  PetscBinarySeek(fd, 0, PETSC_BINARY_SEEK_END, &offset);

  MatView(mat_, viewer);

  PetscViewerDestroy(&viewer);

  return offset;
}

void MyMat::initialize_by_owned()
{
  mat_ = mat_owned_;
  ownership_info_ = MyMat::Ownership(mat_);
}

void SparseMatrixNonzeroStruct::loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*)> process) const
{
  for (MyIndexType i = 0; i < get_row_ptrs().size() - 1; ++i) {
    MyIndexType head_index = get_row_ptrs()[i];
    MyIndexType tail_index = get_row_ptrs()[i + 1];

    MyIndexType n_element = tail_index - head_index;

    process(i, n_element, &get_columns()[head_index]);
  }
}


#endif


} // namespace cppoqss
