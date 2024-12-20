/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/base/cuda.h>
#include <deal.II/base/cuda_size.h>
#include <deal.II/lac/cuda_kernels.h>
#include <deal.II/lac/read_write_vector.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/cuda_vector.h>

namespace ExaDG
{
#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8

template<typename Number>
__global__ void
vec_invert(Number * v, const dealii::types::global_dof_index N)
{
  const auto idx_base = threadIdx.x + blockIdx.x * (blockDim.x * CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c)
  {
    const auto idx = idx_base + c * BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v[idx] = (abs(v[idx]) < 1e-10) ? 1.0 : 1.0 / v[idx];
  }
}

template<typename VectorType>
void
vector_invert(VectorType & vec)
{
  if(vec.locally_owned_size() == 0)
    return;

  const int nblocks =
    1 + (vec.locally_owned_size() - 1) / (CHUNKSIZE_ELEMWISE_OP * BKSIZE_ELEMWISE_OP);
  vec_invert<typename VectorType::value_type>
    <<<nblocks, BKSIZE_ELEMWISE_OP>>>(vec.get_values(), vec.locally_owned_size());
  AssertCudaKernel();
}



template<typename Number>
CudaVector<Number>::CudaVector(const CudaVector<Number> & V)
  : dealii::Subscriptor(),
    val(dealii::Utilities::CUDA::allocate_device_data<Number>(V.n_elements),
        dealii::Utilities::CUDA::delete_device_data<Number>),
    n_elements(V.n_elements)
{
  // Copy the values.
  const cudaError_t error_code =
    cudaMemcpy(val.get(), V.val.get(), n_elements * sizeof(Number), cudaMemcpyDeviceToDevice);
  AssertCuda(error_code);
}

template<typename Number>
CudaVector<Number>::CudaVector()
  : dealii::Subscriptor(),
    val(nullptr, dealii::Utilities::CUDA::delete_device_data<Number>),
    n_elements(0)
{
}

template<typename Number>
CudaVector<Number>::~CudaVector()
{
}

template<typename Number>
void
CudaVector<Number>::reinit(const size_type n, const bool omit_zeroing_entries)
{
  // Resize the underlying array if necessary
  if(n == 0)
    val.reset();
  else if(n != n_elements)
    val.reset(dealii::Utilities::CUDA::allocate_device_data<Number>(n));

  // If necessary set the elements to zero
  if(omit_zeroing_entries == false)
  {
    const cudaError_t error_code = cudaMemset(val.get(), 0, n * sizeof(Number));
    AssertCuda(error_code);
  }
  n_elements = n;
}

template<typename Number>
void
CudaVector<Number>::import(const dealii::LinearAlgebra::ReadWriteVector<Number> & V,
                           dealii::VectorOperation::values                        operation)
{
  if(operation == dealii::VectorOperation::insert)
  {
    const cudaError_t error_code =
      cudaMemcpy(val.get(), V.begin(), n_elements * sizeof(Number), cudaMemcpyHostToDevice);
    AssertCuda(error_code);
  }
  else if(operation == dealii::VectorOperation::add)
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
  else
    AssertThrow(false, dealii::ExcNotImplemented());
}

template<typename Number>
std::size_t
CudaVector<Number>::memory_consumption() const
{
  std::size_t memory = sizeof(*this);
  memory += sizeof(Number) * static_cast<std::size_t>(n_elements);

  return memory;
}


template class CudaVector<dealii::types::global_dof_index>;
template class CudaVector<float>;
template class CudaVector<double>;

} // namespace ExaDG
