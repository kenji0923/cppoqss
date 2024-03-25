#ifndef ARITHMETIC_H
#define ARITHMETIC_H


#include <array>
#include <filesystem>
#include <functional>
#include <sys/types.h>
#include <tuple>
#include <unordered_set>
#include <vector>

#define USE_PETSC

#ifdef USE_PETSC
#include <petscmat.h>
#endif

#include "mpi_helper.h"


namespace cppoqss {


#ifdef USE_PETSC


typedef PetscInt MyIndexType;
typedef PetscScalar MyElementType;

struct MyRowCopied
{
  MyIndexType n_col;
  std::vector<MyIndexType> cols;
  std::vector<MyElementType> values;
};

struct MyRow
{
  MyRowCopied copy() const;

  MyIndexType n_col;
  const MyIndexType* cols_ptr;
  const MyElementType* values_ptr;
};

class MyVec : IsRequireMPI
{
public:
    struct Ownership
    {
	Ownership() { }
	Ownership(const Vec& vec);

	MyIndexType n_dim;
	MyIndexType n;
	std::array<MyIndexType, 2> range;

	std::vector<MyIndexType> indices;
    };

    MyVec(const MyIndexType n_dim);
    MyVec(const Vec vec);
    MyVec(const std::filesystem::path& read_path, const off_t offset);
    MyVec(const MyVec& rh);
    MyVec(MyVec&& rh);
    ~MyVec();

    MyVec& operator=(const MyVec& rh);
    MyVec& operator=(MyVec&& rh);

    MyVec duplicate() const;
    void reset() { VecSet(vec_, 0.0); }

    int set_all(const MyElementType value) { return VecSet(vec_, value); }

    MyVec& add_ax(const MyElementType a, const MyVec& x);
    MyVec& add_to_all_elements(const MyElementType add_value);
    MyVec& pointwise_divide(const MyVec& denominator);
    MyVec& scale(const MyElementType scale_factor);
    MyVec& convert_to_abs();

    MyIndexType get_n_dim() const { return ownership_info_.n_dim; }
    MyIndexType get_n_ownership() const { return ownership_info_.n; }
    const auto& get_ownership_range() const { return ownership_info_.range; }

    MyRow get_row() const;
    void restore_row(MyRow& row) const;

    /**
     * Accepts (index, value) -> void
     */
    void loop_over_elements(std::function<void(const MyIndexType, const MyElementType)> process) const;

    void set_values(const MyIndexType n_element, const MyIndexType* indices_ptr, const MyElementType* values_ptr);
    void add_values(const MyIndexType n_element, const MyIndexType* indices_ptr, const MyElementType* values_ptr);

    void begin_assembly() { VecAssemblyBegin(vec_); }
    void end_assembly() { VecAssemblyEnd(vec_); }

    void write_to_file(const std::filesystem::path& path) const;
    off_t append_to_file(const std::filesystem::path& path) const;

private:
    void initialize_by_owned();

    Vec vec_;
    Vec vec_owned_;
    Ownership ownership_info_;
};

/**
 * TODO: hermite check
 */
class MyMat : IsRequireMPI
{
public:
    struct Ownership
    {
	Ownership() { }
	Ownership(const Mat& mat);

	MyIndexType n_dim;
	MyIndexType n_row;
	std::array<MyIndexType, 2> range_row;
	MyIndexType n_col;
	std::array<MyIndexType, 2> range_col;
    };

    struct NonzeroNumbers
    {
	NonzeroNumbers(const MyIndexType n_row) : diag(n_row), nondiag(n_row) { }
	NonzeroNumbers(std::vector<MyIndexType>&& diag, std::vector<MyIndexType>&& nondiag) : diag(std::move(diag)), nondiag(std::move(nondiag)) { }

	std::vector<MyIndexType> diag;
	std::vector<MyIndexType> nondiag;
    };

    static std::tuple<MyIndexType, MyIndexType> get_ownership_range_before_preallocation(const MyMat& mat);
    static MyMat MatMatMulti(const MyMat& mat_left, const MyMat& mat_right);
    static MyMat HermitianTranspose(const MyMat& mat_source);

    static MatType DefaultMatType;

    MyMat(const MyIndexType n_dim, bool is_hermite=false);
    MyMat(const Mat mat);
    MyMat(const std::filesystem::path& read_path, const off_t offset);
    MyMat(const MyMat& rh);
    MyMat(MyMat&& rh);
    ~MyMat();

    MyMat& operator=(const MyMat& rh);
    MyMat& operator=(MyMat&& rh);

    void reset();

    MyMat duplicate() const;

    MyVec get_diagonal() const;

    MyIndexType get_n_dim() const { return ownership_info_.n_dim; }
    MyIndexType get_n_ownership_row() const { return ownership_info_.n_row; }
    const auto& get_ownership_range_row() const { return ownership_info_.range_row; }
    MyIndexType get_n_ownership_col() const { return ownership_info_.n_col; }
    const auto& get_ownership_range_col() const { return ownership_info_.range_col; }

    MyIndexType get_number_of_nonzeros() const;
    NonzeroNumbers get_number_of_nonzeros_structured() const;

    MyRow get_row(const MyIndexType i) const;
    void restore_row(const MyIndexType i, MyRow& row) const;

    void set_values(const MyIndexType n_row, const MyIndexType* ptr_rows, const MyIndexType n_col, const MyIndexType* ptr_cols, const MyElementType* ptr_elements);
    void add_values(const MyIndexType n_row, const MyIndexType* ptr_rows, const MyIndexType n_col, const MyIndexType* ptr_cols, const MyElementType* ptr_elements);

    MyMat& add_ax(const MyElementType a, const MyMat& x);
    MyMat& add_to_all_elements(const MyElementType add_value);
    MyMat& scale(const MyElementType scale_factor);

    void set_preallocation(const NonzeroNumbers& nonzero_numbers);
    void set_preallocation(const MyIndexType n_diag_nonzero, const MyIndexType* array_diag_nonzero, const MyIndexType n_nondiag_nonzero, const MyIndexType* array_nondiag_nonzero);

    /**
     * Accepts callable object whose type is (i, j) -> void.
     */
    void loop_over_elements(std::function<void(const MyIndexType, const MyIndexType)> process);

    /**
     * Accepts callable object whose type is (i, n_col, cols_ptr, values_ptr) -> void.
     */
    void loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*, const MyElementType*)> process) const;

    /**
     * Accepts callable object whose type is (i, j, element) -> void.
     */
    void loop_over_nonzero_elements(std::function<void(const MyIndexType, const MyIndexType, const MyElementType)> process) const;

    /**
     * Accepts callable object whose type is (i, j, element) -> modified_element.
     */
    void modify_nonzero_elements(std::function<MyElementType(const MyIndexType, const MyIndexType, const MyElementType)> modifier);

    /**
     * Accepts callable object whose type is (i, j) -> bool.
     */
    NonzeroNumbers get_nonzero_numbers(std::function<bool(const MyIndexType, const MyIndexType)> checker_gives_true_if_nonzero);

    void begin_final_assembly() { MatAssemblyBegin(mat_, MAT_FINAL_ASSEMBLY); }
    void end_final_assembly() { MatAssemblyEnd(mat_, MAT_FINAL_ASSEMBLY); }

    void write_to_file(const std::filesystem::path& path) const;
    off_t append_to_file(const std::filesystem::path& path) const;

private:
    void initialize_by_owned();

    Mat mat_;
    Mat mat_owned_;
    Ownership ownership_info_;

    bool is_hermite_;
};

#endif

class SparseMatrixNonzeroStruct
{
public:
  void push_back_element(const size_t col) { columns_.push_back(col); }
  void push_back_row() { row_ptrs_.push_back(columns_.size()); }

  MyIndexType get_n_row() const { return row_ptrs_.size(); }
  const std::vector<MyIndexType>& get_columns() const { return columns_; }
  const std::vector<size_t>& get_row_ptrs() const { return row_ptrs_; }

  void loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*)> process) const;

private:
  std::vector<MyIndexType> columns_;
  std::vector<size_t> row_ptrs_ { 0 };
};

template<class T>
class SparseMatrix
{
public:
  void push_back_element(const size_t col, const T& element) { nonzero_struct_.push_back_element(col); elements_.push_back(element); }
  void push_back_row() { nonzero_struct_.push_back_row(); }

  void loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*, const MyElementType*)> process);

private:
  std::vector<T> elements_;
  SparseMatrixNonzeroStruct nonzero_struct_;
};

template<class T>

void SparseMatrix<T>::loop_by_row(std::function<void(const MyIndexType, const MyIndexType, const MyIndexType*, const MyElementType*)> process)
{
  for (MyIndexType i = 0; i <nonzero_struct_.get_row_ptrs().size() - 1; ++i) {
    MyIndexType head_index = nonzero_struct_.get_row_ptrs()[i];
    MyIndexType tail_index = nonzero_struct_.get_row_ptrs()[i + 1];

    MyIndexType n_element = tail_index - head_index;

    process(i, n_element, &nonzero_struct_.get_columns()[head_index], &elements_[head_index]);
  }
}


} // cppoqss
  

#endif
