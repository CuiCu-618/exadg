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
#include <deal.II/multigrid/mg_transfer_internal.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/cuda_mg_transfer_h.h>

namespace ExaDG
{
namespace CUDAWrappers
{

enum TransferVariant
{
  PROLONGATION,
  RESTRICTION
};

template<int dim, int fe_degree, typename Number>
class MGTransferHelper
{
protected:
  static constexpr unsigned int n_coarse = fe_degree + 1;
  static constexpr unsigned int n_fine   = fe_degree * 2 + 2;
  static constexpr unsigned int M        = 2;

  Number *                                values;
  const Number *                          weights;
  const Number *                          shape_values;
  const dealii::types::global_dof_index * dof_indices_coarse;
  const dealii::types::global_dof_index * dof_indices_fine;

  __device__
  MGTransferHelper(Number *                                buf,
                   const Number *                          w,
                   const Number *                          shvals,
                   const dealii::types::global_dof_index * idx_coarse,
                   const dealii::types::global_dof_index * idx_fine)
    : values(buf),
      weights(w),
      shape_values(shvals),
      dof_indices_coarse(idx_coarse),
      dof_indices_fine(idx_fine)
  {
  }

  template<TransferVariant transfer_type, int dir>
  __device__ void
  reduce(const Number * my_shvals)
  {
    // multiplicity of large and small size
    constexpr bool prol  = transfer_type == PROLONGATION;
    constexpr auto n_src = prol ? n_coarse : n_fine;

    // in direction of reduction (dir and threadIdx.x respectively), always
    // read from 1 location, and write to M (typically 2). in other
    // directions, either read M or 1 and write same number.
    constexpr auto M1 = prol ? M : 1;
    constexpr auto M2 = prol ? (dir > 0 ? M : 1) : ((dir > 0 || dim < 2) ? 1 : M);
    constexpr auto M3 = prol ? (dir > 1 ? M : 1) : ((dir > 1 || dim < 3) ? 1 : M);

    Number tmp[M1 * M2 * M3];

#pragma unroll
    for(int m3 = 0; m3 < M3; ++m3)
    {
#pragma unroll
      for(int m2 = 0; m2 < M2; ++m2)
      {
#pragma unroll
        for(int m1 = 0; m1 < M1; ++m1)
        {
          tmp[m1 + M1 * (m2 + M2 * m3)] = 0;

          for(int i = 0; i < n_src; ++i)
          {
            const auto x   = i;
            const auto y   = m2 + M2 * threadIdx.y;
            const auto z   = m3 + M3 * threadIdx.z;
            const auto idx = (dir == 0 ? x + n_fine * (y + n_fine * z) :
                              dir == 1 ? y + n_fine * (x + n_fine * z) :
                                         y + n_fine * (z + n_fine * x));

            tmp[m1 + M1 * (m2 + M2 * m3)] += my_shvals[m1 * n_src + i] * values[idx];
          }
        }
      }
    }
    __syncthreads();

#pragma unroll
    for(int m3 = 0; m3 < M3; ++m3)
    {
#pragma unroll
      for(int m2 = 0; m2 < M2; ++m2)
      {
#pragma unroll
        for(int m1 = 0; m1 < M1; ++m1)
        {
          const auto x   = m1 + M1 * threadIdx.x;
          const auto y   = m2 + M2 * threadIdx.y;
          const auto z   = m3 + M3 * threadIdx.z;
          const auto idx = (dir == 0 ? x + n_fine * (y + n_fine * z) :
                            dir == 1 ? y + n_fine * (x + n_fine * z) :
                                       y + n_fine * (z + n_fine * x));


          values[idx] = tmp[m1 + M1 * (m2 + M2 * m3)];
        }
      }
    }
  }
};

template<int dim, int fe_degree, typename Number>
class MGProlongateHelper : public MGTransferHelper<dim, fe_degree, Number>
{
  using MGTransferHelper<dim, fe_degree, Number>::M;
  using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
  using MGTransferHelper<dim, fe_degree, Number>::n_fine;
  using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
  using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
  using MGTransferHelper<dim, fe_degree, Number>::values;
  using MGTransferHelper<dim, fe_degree, Number>::shape_values;
  using MGTransferHelper<dim, fe_degree, Number>::weights;

public:
  __device__
  MGProlongateHelper(Number *                                buf,
                     const Number *                          w,
                     const Number *                          shvals,
                     const dealii::types::global_dof_index * idx_coarse,
                     const dealii::types::global_dof_index * idx_fine)
    : MGTransferHelper<dim, fe_degree, Number>(buf, w, shvals, idx_coarse, idx_fine)
  {
  }

  __device__ void
  run(Number * dst, const Number * src)
  {
    Number my_shvals[M * n_coarse];
    for(int m = 0; m < (threadIdx.x < fe_degree ? M : M); ++m)
      for(int i = 0; i < n_coarse; ++i)
        my_shvals[m * n_coarse + i] = shape_values[(threadIdx.x * M + m) + n_fine * i];

    read_coarse(src);
    __syncthreads();

    this->template reduce<PROLONGATION, 0>(my_shvals);
    __syncthreads();
    if(dim > 1)
    {
      this->template reduce<PROLONGATION, 1>(my_shvals);
      __syncthreads();
      if(dim > 2)
      {
        this->template reduce<PROLONGATION, 2>(my_shvals);
        __syncthreads();
      }
    }

    write_fine(dst);
  }

private:
  __device__ void
  read_coarse(const Number * vec)
  {
    const auto idx = threadIdx.x + n_fine * (threadIdx.y + n_fine * threadIdx.z);
    values[idx]    = vec[dof_indices_coarse[idx]];
  }

  __device__ void
  write_fine(Number * vec) const
  {
    const auto M1 = M;
    const auto M2 = (dim > 1 ? M : 1);
    const auto M3 = (dim > 2 ? M : 1);

    for(int m3 = 0; m3 < M3; ++m3)
      for(int m2 = 0; m2 < M2; ++m2)
        for(int m1 = 0; m1 < M1; ++m1)
        {
          const auto x = (M1 * threadIdx.x + m1);
          const auto y = (M2 * threadIdx.y + m2);
          const auto z = (M3 * threadIdx.z + m3);

          const auto idx = x + n_fine * (y + n_fine * z);
          if(x < n_fine && y < n_fine && z < n_fine)
            atomicAdd(&vec[dof_indices_fine[idx]], values[idx]);
        }
  }
};

template<int dim, int fe_degree, typename Number>
class MGRestrictHelper : public MGTransferHelper<dim, fe_degree, Number>
{
  using MGTransferHelper<dim, fe_degree, Number>::M;
  using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
  using MGTransferHelper<dim, fe_degree, Number>::n_fine;
  using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
  using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
  using MGTransferHelper<dim, fe_degree, Number>::values;
  using MGTransferHelper<dim, fe_degree, Number>::shape_values;
  using MGTransferHelper<dim, fe_degree, Number>::weights;

public:
  __device__
  MGRestrictHelper(Number *                                buf,
                   const Number *                          w,
                   const Number *                          shvals,
                   const dealii::types::global_dof_index * idx_coarse,
                   const dealii::types::global_dof_index * idx_fine)
    : MGTransferHelper<dim, fe_degree, Number>(buf, w, shvals, idx_coarse, idx_fine)
  {
  }

  __device__ void
  run(Number * dst, const Number * src)
  {
    Number my_shvals[n_fine];
    for(int i = 0; i < n_fine; ++i)
      my_shvals[i] = shape_values[threadIdx.x * n_fine + i];

    read_fine(src);
    __syncthreads();

    this->template reduce<RESTRICTION, 0>(my_shvals);
    __syncthreads();
    if(dim > 1)
    {
      this->template reduce<RESTRICTION, 1>(my_shvals);
      __syncthreads();
      if(dim > 2)
      {
        this->template reduce<RESTRICTION, 2>(my_shvals);
        __syncthreads();
      }
    }

    write_coarse(dst);
  }

private:
  __device__ void
  read_fine(const Number * vec)
  {
    const auto M1 = M;
    const auto M2 = (dim > 1 ? M : 1);
    const auto M3 = (dim > 2 ? M : 1);

    for(int m3 = 0; m3 < M3; ++m3)
      for(int m2 = 0; m2 < M2; ++m2)
        for(int m1 = 0; m1 < M1; ++m1)
        {
          const auto x = (M1 * threadIdx.x + m1);
          const auto y = (M2 * threadIdx.y + m2);
          const auto z = (M3 * threadIdx.z + m3);

          const auto idx = x + n_fine * (y + n_fine * z);
          if(x < n_fine && y < n_fine && z < n_fine)
            values[idx] = vec[dof_indices_fine[idx]];
        }
  }

  __device__ void
  write_coarse(Number * vec) const
  {
    const auto idx = threadIdx.x + n_fine * (threadIdx.y + n_fine * threadIdx.z);

    atomicAdd(&vec[dof_indices_coarse[idx]], values[idx]);
  }
};

namespace internal
{
extern __shared__ double shmem_d[];
extern __shared__ float  shmem_f[];

template<typename Number>
__device__ inline Number *
get_shared_mem_ptr();

template<>
__device__ inline double *
get_shared_mem_ptr()
{
  return shmem_d;
}

template<>
__device__ inline float *
get_shared_mem_ptr()
{
  return shmem_f;
}
} // namespace internal

template<int dim, int degree, typename loop_body, typename Number>
__global__ void __launch_bounds__(1024, 1)
  mg_kernel(Number *                                dst,
            const Number *                          src,
            const Number *                          weights,
            const Number *                          shape_values,
            const dealii::types::global_dof_index * dof_indices_coarse,
            const dealii::types::global_dof_index * dof_indices_fine,
            const dealii::types::global_dof_index * child_offset_in_parent,
            const unsigned int                      n_child_cell_dofs)
{
  const auto coarse_cell   = blockIdx.x;
  const auto coarse_offset = child_offset_in_parent[coarse_cell];

  loop_body body(internal::get_shared_mem_ptr<Number>(),
                 weights, // + coarse_cell * pow(3, dim),
                 shape_values,
                 dof_indices_coarse + coarse_offset,
                 dof_indices_fine + coarse_cell * n_child_cell_dofs);

  body.run(dst, src);
}

template<int dim, typename Number>
template<template<int, int, typename> class loop_body, int degree>
void
MGTransferH<dim, Number>::coarse_cell_loop(const unsigned int fine_level,
                                           VectorType &       dst,
                                           const VectorType & src) const
{
  const auto n_fine_size      = std::pow(degree * 2 + 2, dim) * sizeof(Number);
  const auto n_coarse_dofs_1d = degree + 1;

  const auto n_coarse_cells = n_owned_level_cells[fine_level - 1];

  // kernel parameters
  dim3 bk_dim(n_coarse_dofs_1d, (dim > 1) ? n_coarse_dofs_1d : 1, (dim > 2) ? n_coarse_dofs_1d : 1);

  dim3 gd_dim(n_coarse_cells);

  AssertCuda(cudaFuncSetAttribute(mg_kernel<dim, degree, loop_body<dim, degree, Number>, Number>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  n_fine_size));

  if(n_coarse_cells > 0)
    mg_kernel<dim, degree, loop_body<dim, degree, Number>><<<gd_dim, bk_dim, n_fine_size>>>(
      dst.get_values(),
      src.get_values(),
      weights_on_refined[fine_level - 1].get_values(), // only has fine-level entries
      prolongation_matrix_1d.get_values(),
      level_dof_indices[fine_level - 1].get_values(),
      level_dof_indices[fine_level].get_values(),
      child_offset_in_parent[fine_level - 1].get_values(), // on coarse level
      n_child_cell_dofs);

  AssertCudaKernel();
}

#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8

template<bool add, typename Number, typename Number2>
__global__ void
vec_equ(Number * dst, const Number2 * src, const dealii::types::global_dof_index N)
{
  const auto idx_base = threadIdx.x + blockIdx.x * (blockDim.x * CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c)
  {
    const auto idx = idx_base + c * BKSIZE_ELEMWISE_OP;
    if(idx < N)
    {
      if(add)
        dst[idx] += src[idx];
      else
        dst[idx] = src[idx];
    }
  }
}

template<bool add, typename VectorType, typename VectorType2>
void
plain_copy(VectorType & dst, const VectorType2 & src)
{
  if(src.locally_owned_size() == 0)
    return;

  if(dst.locally_owned_size() != src.locally_owned_size())
  {
    dst.reinit(src.locally_owned_size(), true);
  }

  const int nblocks =
    1 + (src.locally_owned_size() - 1) / (CHUNKSIZE_ELEMWISE_OP * BKSIZE_ELEMWISE_OP);

  vec_equ<add, typename VectorType::value_type, typename VectorType2::value_type>
    <<<nblocks, BKSIZE_ELEMWISE_OP>>>(dst.get_values(), src.get_values(), src.locally_owned_size());
  AssertCudaKernel();
}



template<typename Number, typename Number2, bool add>
__global__ void
copy_with_indices_kernel(Number *                                dst,
                         Number2 *                               src,
                         const dealii::types::global_dof_index * dst_indices,
                         const dealii::types::global_dof_index * src_indices,
                         dealii::types::global_dof_index         n)
{
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n)
  {
    if(add)
      dst[dst_indices[i]] += src[src_indices[i]];
    else
      dst[dst_indices[i]] = src[src_indices[i]];
  }
}

template<typename Number, typename Number2, bool add = false>
void
copy_with_indices(
  dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> &        dst,
  const dealii::LinearAlgebra::distributed::Vector<Number2, dealii::MemorySpace::CUDA> & src,
  const CudaVector<dealii::types::global_dof_index> & dst_indices,
  const CudaVector<dealii::types::global_dof_index> & src_indices)
{
  const auto n         = dst_indices.size();
  const auto blocksize = 256;
  const dim3 block_dim = dim3(blocksize);
  const dim3 grid_dim  = dim3(1 + (n - 1) / blocksize);
  copy_with_indices_kernel<Number, Number2, add><<<grid_dim, block_dim>>>(
    dst.get_values(), src.get_values(), dst_indices.get_values(), src_indices.get_values(), n);
  AssertCudaKernel();
}


template<int dim, typename Number>
MGTransferH<dim, Number>::MGTransferH(
  std::map<unsigned int, unsigned int> level_to_triangulation_level_map,
  dealii::DoFHandler<dim> const &      dof_handler)
  : underlying_operator(0),
    level_to_triangulation_level_map(level_to_triangulation_level_map),
    dof_handler(dof_handler),
    fe_degree(0),
    element_is_continuous(false),
    n_components(0),
    n_child_cell_dofs(0)
{
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::initialize_constraints(
  const dealii::MGConstrainedDoFs & mg_constrained_dofs)
{
  this->mg_constrained_dofs = &mg_constrained_dofs;
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::build(
  const dealii::DoFHandler<dim, dim> & dof_handler,
  const std::vector<std::shared_ptr<const dealii::Utilities::MPI::Partitioner>> &
    external_partitioners)
{
  Assert(dof_handler.has_level_dofs(),
         dealii::ExcMessage("The underlying DoFHandler object has not had its "
                            "distribute_mg_dofs() function called, but this is a prerequisite "
                            "for multigrid transfers. You will need to call this function, "
                            "probably close to where you already call distribute_dofs()."));

  /**
   * Only global refinement so far, just plain copy. Uncomment for adaptice
   * refinement.
   */
  fill_copy_indices(dof_handler);

  const unsigned int n_levels = dof_handler.get_triangulation().n_global_levels();

  vector_partitioners.resize(0, n_levels - 1);
  for(unsigned int level = 0; level <= ghosted_level_vector.max_level(); ++level)
    vector_partitioners[level] = ghosted_level_vector[level].get_partitioner();

  std::vector<std::vector<Number>>                                weights_host;
  std::vector<std::vector<unsigned int>>                          level_dof_indices_host;
  std::vector<std::vector<std::pair<unsigned int, unsigned int>>> parent_child_connect;

  // std::vector<Table<2, unsigned int>> copy_indices_global_mine;
  // MGLevelObject<LinearAlgebra::distributed::Vector<Number>>
  //   ghosted_level_vector;
  std::vector<std::vector<std::vector<unsigned short>>> dirichlet_indices_host;

  ghosted_level_vector.resize(0, n_levels - 1);

  vector_partitioners.resize(0, n_levels - 1);
  for(unsigned int level = 0; level <= ghosted_level_vector.max_level(); ++level)
    vector_partitioners[level] = ghosted_level_vector[level].get_partitioner();

  // WARN: setup_transfer() only works with "unsigned int"
  dealii::internal::MGTransfer::ElementInfo<Number> elem_info;
  dealii::internal::MGTransfer::setup_transfer<dim, Number>(dof_handler,
                                                            this->mg_constrained_dofs,
                                                            external_partitioners,
                                                            elem_info,
                                                            level_dof_indices_host,
                                                            parent_child_connect,
                                                            n_owned_level_cells,
                                                            dirichlet_indices_host,
                                                            weights_host,
                                                            copy_indices_global_mine_host,
                                                            vector_partitioners);

  // unpack element info data
  fe_degree             = elem_info.fe_degree;
  element_is_continuous = elem_info.element_is_continuous;
  n_components          = elem_info.n_components;
  n_child_cell_dofs     = elem_info.n_child_cell_dofs;

  //---------------------------------------------------------------------------
  // transfer stuff from host to device
  //---------------------------------------------------------------------------
  copy_to_device(prolongation_matrix_1d, elem_info.prolongation_matrix_1d);

  level_dof_indices.resize(n_levels);

  for(unsigned int l = 0; l < n_levels; l++)
  {
    copy_to_device(level_dof_indices[l], level_dof_indices_host[l]);
  }

  weights_on_refined.resize(n_levels - 1);
  // for (unsigned int l = 0; l < n_levels - 1; l++)
  //   {
  //     copy_to_device(weights_on_refined[l], weights_host[l]);
  //   }

  child_offset_in_parent.resize(n_levels - 1);
  std::vector<dealii::types::global_dof_index> offsets;

  for(unsigned int l = 0; l < n_levels - 1; l++)
  {
    offsets.resize(n_owned_level_cells[l]);

    for(unsigned int c = 0; c < n_owned_level_cells[l]; ++c)
    {
      const auto shift = dealii::internal::MGTransfer::compute_shift_within_children<dim>(
        parent_child_connect[l][c].second, fe_degree + 1 - element_is_continuous, fe_degree);
      offsets[c] = parent_child_connect[l][c].first * n_child_cell_dofs + shift;
    }

    copy_to_device(child_offset_in_parent[l], offsets);
  }

  std::vector<dealii::types::global_dof_index> dirichlet_index_vector;
  dirichlet_indices.resize(n_levels);
  if(this->mg_constrained_dofs != nullptr && mg_constrained_dofs->have_boundary_indices())
  {
    for(unsigned int l = 0; l < n_levels; l++)
    {
      mg_constrained_dofs->get_boundary_indices(l).fill_index_vector(dirichlet_index_vector);
      copy_to_device(dirichlet_indices[l], dirichlet_index_vector);
    }
  }
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::prolongate_and_add(unsigned int const to_level_,
                                             VectorType &       dst,
                                             VectorType const & src) const
{
  auto to_level = level_to_triangulation_level_map[to_level_];
  Assert((to_level >= 1) && (to_level <= level_dof_indices.size()),
         dealii::ExcIndexRange(to_level, 1, level_dof_indices.size() + 1));

  const bool src_inplace =
    src.get_partitioner().get() == this->vector_partitioners[to_level - 1].get();
  if(src_inplace == false)
  {
    if(this->ghosted_level_vector[to_level - 1].get_partitioner().get() !=
       this->vector_partitioners[to_level - 1].get())
      this->ghosted_level_vector[to_level - 1].reinit(this->vector_partitioners[to_level - 1]);
    this->ghosted_level_vector[to_level - 1].copy_locally_owned_data_from(src);
  }

  const bool dst_inplace = dst.get_partitioner().get() == this->vector_partitioners[to_level].get();
  if(dst_inplace == false)
  {
    if(this->ghosted_level_vector[to_level].get_partitioner().get() !=
       this->vector_partitioners[to_level].get())
      this->ghosted_level_vector[to_level].reinit(this->vector_partitioners[to_level]);
    AssertDimension(this->ghosted_level_vector[to_level].locally_owned_size(),
                    dst.locally_owned_size());
    this->ghosted_level_vector[to_level] = 0.;
  }

  const VectorType & src_vec = src_inplace ? src : this->ghosted_level_vector[to_level - 1];
  VectorType &       dst_vec = dst_inplace ? dst : this->ghosted_level_vector[to_level];

  src_vec.update_ghost_values();

  if(fe_degree == 1)
    coarse_cell_loop<MGProlongateHelper, 1>(to_level, dst_vec, src_vec);
  else if(fe_degree == 2)
    coarse_cell_loop<MGProlongateHelper, 2>(to_level, dst_vec, src_vec);
  else if(fe_degree == 3)
    coarse_cell_loop<MGProlongateHelper, 3>(to_level, dst_vec, src_vec);
  else if(fe_degree == 4)
    coarse_cell_loop<MGProlongateHelper, 4>(to_level, dst_vec, src_vec);
  else if(fe_degree == 5)
    coarse_cell_loop<MGProlongateHelper, 5>(to_level, dst_vec, src_vec);
  else if(fe_degree == 6)
    coarse_cell_loop<MGProlongateHelper, 6>(to_level, dst_vec, src_vec);
  else if(fe_degree == 7)
    coarse_cell_loop<MGProlongateHelper, 7>(to_level, dst_vec, src_vec);
  else if(fe_degree == 8)
    coarse_cell_loop<MGProlongateHelper, 8>(to_level, dst_vec, src_vec);
  else if(fe_degree == 9)
    coarse_cell_loop<MGProlongateHelper, 9>(to_level, dst_vec, src_vec);
  else if(fe_degree == 10)
    coarse_cell_loop<MGProlongateHelper, 10>(to_level, dst_vec, src_vec);
  else
    AssertThrow(false, dealii::ExcNotImplemented("Only degrees 1 through 10 implemented."));

  dst_vec.compress(dealii::VectorOperation::add);
  if(dst_inplace == false)
    dst += dst_vec;

  if(src_inplace == true)
    src.zero_out_ghost_values();
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::restrict_and_add(unsigned int const from_level_,
                                           VectorType &       dst,
                                           VectorType const & src) const
{
  auto from_level = level_to_triangulation_level_map[from_level_];

  Assert((from_level >= 1) && (from_level <= level_dof_indices.size()),
         dealii::ExcIndexRange(from_level, 1, level_dof_indices.size() + 1));

  const bool src_inplace =
    src.get_partitioner().get() == this->vector_partitioners[from_level].get();
  if(src_inplace == false)
  {
    if(this->ghosted_level_vector[from_level].get_partitioner().get() !=
       this->vector_partitioners[from_level].get())
      this->ghosted_level_vector[from_level].reinit(this->vector_partitioners[from_level]);
    this->ghosted_level_vector[from_level].copy_locally_owned_data_from(src);
  }

  const bool dst_inplace =
    dst.get_partitioner().get() == this->vector_partitioners[from_level - 1].get();
  if(dst_inplace == false)
  {
    if(this->ghosted_level_vector[from_level - 1].get_partitioner().get() !=
       this->vector_partitioners[from_level - 1].get())
      this->ghosted_level_vector[from_level - 1].reinit(this->vector_partitioners[from_level - 1]);
    AssertDimension(this->ghosted_level_vector[from_level - 1].locally_owned_size(),
                    dst.locally_owned_size());
    this->ghosted_level_vector[from_level - 1] = 0.;
  }

  const VectorType & src_vec = src_inplace ? src : this->ghosted_level_vector[from_level];
  VectorType &       dst_vec = dst_inplace ? dst : this->ghosted_level_vector[from_level - 1];

  src_vec.update_ghost_values();

  if(fe_degree == 1)
    coarse_cell_loop<MGRestrictHelper, 1>(from_level, dst_vec, src_vec);
  else if(fe_degree == 2)
    coarse_cell_loop<MGRestrictHelper, 2>(from_level, dst_vec, src_vec);
  else if(fe_degree == 3)
    coarse_cell_loop<MGRestrictHelper, 3>(from_level, dst_vec, src_vec);
  else if(fe_degree == 4)
    coarse_cell_loop<MGRestrictHelper, 4>(from_level, dst_vec, src_vec);
  else if(fe_degree == 5)
    coarse_cell_loop<MGRestrictHelper, 5>(from_level, dst_vec, src_vec);
  else if(fe_degree == 6)
    coarse_cell_loop<MGRestrictHelper, 6>(from_level, dst_vec, src_vec);
  else if(fe_degree == 7)
    coarse_cell_loop<MGRestrictHelper, 7>(from_level, dst_vec, src_vec);
  else if(fe_degree == 8)
    coarse_cell_loop<MGRestrictHelper, 8>(from_level, dst_vec, src_vec);
  else if(fe_degree == 9)
    coarse_cell_loop<MGRestrictHelper, 9>(from_level, dst_vec, src_vec);
  else if(fe_degree == 10)
    coarse_cell_loop<MGRestrictHelper, 10>(from_level, dst_vec, src_vec);
  else
    AssertThrow(false, dealii::ExcNotImplemented("Only degrees 1 through 10 implemented."));

  dst_vec.compress(dealii::VectorOperation::add);
  if(dst_inplace == false)
    dst += dst_vec;

  if(src_inplace == true)
    src.zero_out_ghost_values();
}

template<typename Number>
__global__ void
set_mg_constrained_dofs_kernel(Number *                                vec,
                               const dealii::types::global_dof_index * indices,
                               dealii::types::global_dof_index         len,
                               Number                                  val)
{
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < len)
  {
    vec[indices[idx]] = val;
  }
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::set_mg_constrained_dofs(VectorType & vec,
                                                  unsigned int level,
                                                  Number       val) const
{
  const auto len = dirichlet_indices[level].size();

  if(len > 0)
  {
    const auto bksize  = 256;
    const auto nblocks = (len - 1) / bksize + 1;
    dim3       bk_dim(bksize);
    dim3       gd_dim(nblocks);

    set_mg_constrained_dofs_kernel<<<gd_dim, bk_dim>>>(vec.get_values(),
                                                       dirichlet_indices[level].get_values(),
                                                       len,
                                                       val);
    AssertCudaKernel();
  }
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::interpolate(unsigned int const level_in,
                                      VectorType &       dst,
                                      VectorType const & src) const
{
  AssertThrow(false, dealii::ExcMessage("TODO"));
}

template<int dim, typename Number>
template<typename VectorType2>
void
MGTransferH<dim, Number>::copy_to_mg(dealii::MGLevelObject<VectorType> & dst,
                                     VectorType2 const &                 src) const
{
  AssertThrow(underlying_operator != 0, dealii::ExcNotInitialized());

  for(unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

  AssertIndexRange(dst.max_level(), dof_handler.get_triangulation().n_global_levels());
  AssertIndexRange(dst.min_level(), dst.max_level() + 1);

  VectorType & this_ghosted_global_vector   = ghosted_global_vector;
  auto &       this_copy_indices            = copy_indices;
  auto &       this_copy_indices_level_mine = copy_indices_level_mine;

  for(unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    if(dst[level].size() != dof_handler.n_dofs(level) ||
       dst[level].locally_owned_size() != dof_handler.locally_owned_mg_dofs(level).n_elements())
    {
      // In case a ghosted level vector has been initialized, we can
      // simply use that as a template for the vector partitioning. If
      // not, we resort to the locally owned range of the dof handler.
      if(level <= ghosted_level_vector.max_level() &&
         ghosted_level_vector[level].size() == dof_handler.n_dofs(level))
        dst[level].reinit(ghosted_level_vector[level], false);
      else
        dst[level].reinit(dof_handler.locally_owned_mg_dofs(level), dof_handler.get_communicator());
    }
    else if((perform_plain_copy == false && perform_renumbered_plain_copy == false) ||
            level != dst.max_level())
      dst[level] = 0;

  if(perform_plain_copy)
  {
    // In this case, we can simply copy the local range.
    AssertDimension(dst[dst.max_level()].locally_owned_size(), src.locally_owned_size());

    plain_copy<false>(dst[dst.max_level()], src);

    return;
  }
  else if(perform_renumbered_plain_copy)
  {
  }

  std::cout << "Warning! Non-plain copy encourted! \n";

  // copy the source vector to the temporary vector that we hold for the
  // purpose of data exchange
  // this_ghosted_global_vector = src;
  plain_copy<false>(this_ghosted_global_vector, src);
  this_ghosted_global_vector.update_ghost_values();

  for(unsigned int level = dst.max_level() + 1; level != dst.min_level();)
  {
    --level;
    auto & dst_level = dst[level];

    copy_with_indices(dst_level,
                      this_ghosted_global_vector,
                      this_copy_indices[level].level_indices,
                      this_copy_indices[level].global_indices);

    copy_with_indices(dst_level,
                      this_ghosted_global_vector,
                      this_copy_indices_level_mine[level].level_indices,
                      this_copy_indices_level_mine[level].global_indices);

    dst_level.compress(dealii::VectorOperation::insert);
  }
}


template<int dim, typename Number>
template<typename VectorType2>
void
MGTransferH<dim, Number>::copy_from_mg(VectorType2 &                             dst,
                                       const dealii::MGLevelObject<VectorType> & src) const
{
  (void)dof_handler;
  AssertIndexRange(src.max_level(), dof_handler.get_triangulation().n_global_levels());
  AssertIndexRange(src.min_level(), src.max_level() + 1);

  if(perform_plain_copy)
  {
    AssertDimension(dst.locally_owned_size(), src[src.max_level()].locally_owned_size());
    plain_copy<false>(dst, src[src.max_level()]);
    return;
  }
  else if(perform_renumbered_plain_copy)
  {
  }

  std::cout << "Warning! Non-plain copy encourted! \n";

  dst = 0;
  for(unsigned int level = src.min_level(); level <= src.max_level(); ++level)
  {
    // the ghosted vector should already have the correct local size (but
    // different parallel layout)
    if(ghosted_level_vector[level].size() > 0)
      AssertDimension(ghosted_level_vector[level].locally_owned_size(),
                      src[level].locally_owned_size());

    // the first time around, we copy the source vector to the temporary
    // vector that we hold for the purpose of data exchange
    VectorType & ghosted_vector = ghosted_level_vector[level];

    if(ghosted_level_vector[level].size() > 0)
      ghosted_vector = src[level];

    const auto ghosted_vector_ptr =
      (ghosted_level_vector[level].size() > 0) ? &ghosted_vector : &src[level];

    ghosted_vector_ptr->update_ghost_values();

    copy_with_indices(dst,
                      *ghosted_vector_ptr,
                      copy_indices[level].global_indices,
                      copy_indices[level].level_indices);

    copy_with_indices(dst,
                      *ghosted_vector_ptr,
                      copy_indices_global_mine[level].global_indices,
                      copy_indices_global_mine[level].level_indices);
  }
  dst.compress(dealii::VectorOperation::insert);
}

template<int dim, typename Number>
template<typename VectorType1, typename VectorType2>
void
MGTransferH<dim, Number>::copy_to_device(VectorType1 & device, const VectorType2 & host)
{
  dealii::LinearAlgebra::ReadWriteVector<typename VectorType1::value_type> rw_vector(host.size());
  device.reinit(host.size());
  for(dealii::types::global_dof_index i = 0; i < host.size(); ++i)
    rw_vector[i] = host[i];
  device.import(rw_vector, dealii::VectorOperation::insert);
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::fill_copy_indices(const dealii::DoFHandler<dim> & dof_handler)
{
  const MPI_Comm mpi_communicator = dof_handler.get_communicator();

  // fill_internal
  std::vector<
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>>>
    my_copy_indices;
  std::vector<
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>>>
    my_copy_indices_global_mine;
  std::vector<
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>>>
    my_copy_indices_level_mine;

  dealii::internal::MGTransfer::fill_copy_indices(dof_handler,
                                                  mg_constrained_dofs,
                                                  my_copy_indices,
                                                  my_copy_indices_global_mine,
                                                  my_copy_indices_level_mine);

  const unsigned int nlevels = dof_handler.get_triangulation().n_global_levels();

  dealii::IndexSet                             index_set(dof_handler.locally_owned_dofs().size());
  std::vector<dealii::types::global_dof_index> accessed_indices;
  ghosted_level_vector.resize(0, nlevels - 1);
  std::vector<dealii::IndexSet> level_index_set(nlevels);
  for(unsigned int l = 0; l < nlevels; ++l)
  {
    for(const auto & indices : my_copy_indices_level_mine[l])
      accessed_indices.push_back(indices.first);
    std::vector<dealii::types::global_dof_index> accessed_level_indices;
    for(const auto & indices : my_copy_indices_global_mine[l])
      accessed_level_indices.push_back(indices.second);
    std::sort(accessed_level_indices.begin(), accessed_level_indices.end());
    level_index_set[l].set_size(dof_handler.locally_owned_mg_dofs(l).size());
    level_index_set[l].add_indices(accessed_level_indices.begin(), accessed_level_indices.end());
    level_index_set[l].compress();
    ghosted_level_vector[l].reinit(dof_handler.locally_owned_mg_dofs(l),
                                   level_index_set[l],
                                   mpi_communicator);
  }
  std::sort(accessed_indices.begin(), accessed_indices.end());
  index_set.add_indices(accessed_indices.begin(), accessed_indices.end());
  index_set.compress();
  ghosted_global_vector.reinit(dof_handler.locally_owned_dofs(), index_set, mpi_communicator);

  // localize the copy indices for faster access. Since all access will be
  // through the ghosted vector in 'data', we can use this (much faster)
  // option
  copy_indices.resize(nlevels);
  copy_indices_level_mine.resize(nlevels);
  copy_indices_global_mine.resize(nlevels);
  copy_indices_global_mine_host.resize(nlevels);
  for(unsigned int level = 0; level < nlevels; ++level)
  {
    const dealii::Utilities::MPI::Partitioner & global_partitioner =
      *ghosted_global_vector.get_partitioner();
    const dealii::Utilities::MPI::Partitioner & level_partitioner =
      *ghosted_level_vector[level].get_partitioner();

    auto translate_indices =
      [&](const std::vector<std::pair<dealii::types::global_dof_index,
                                      dealii::types::global_dof_index>> & global_copy_indices,
          IndexMapping &                                                  local_copy_indices)
    {
      const dealii::types::global_dof_index nmappings = global_copy_indices.size();
      std::vector<int>                      global_indices(nmappings);
      std::vector<int>                      level_indices(nmappings);

      for(dealii::types::global_dof_index j = 0; j < nmappings; ++j)
      {
        global_indices[j] = global_partitioner.global_to_local(global_copy_indices[j].first);
        level_indices[j]  = level_partitioner.global_to_local(global_copy_indices[j].second);
      }

      copy_to_device(local_copy_indices.global_indices, global_indices);
      copy_to_device(local_copy_indices.level_indices, level_indices);
    };

    // owned-owned case
    translate_indices(my_copy_indices[level], copy_indices[level]);

    // remote-owned case
    translate_indices(my_copy_indices_level_mine[level], copy_indices_level_mine[level]);

    // owned-remote case
    translate_indices(my_copy_indices_global_mine[level], copy_indices_global_mine[level]);

    // copy_indices_global_mine_host
    copy_indices_global_mine_host[level].reinit(2, my_copy_indices_global_mine[level].size());
    for(dealii::types::global_dof_index i = 0; i < my_copy_indices_global_mine[level].size(); ++i)
    {
      copy_indices_global_mine_host[level](0, i) =
        global_partitioner.global_to_local(my_copy_indices_global_mine[level][i].first);
      copy_indices_global_mine_host[level](1, i) =
        level_partitioner.global_to_local(my_copy_indices_global_mine[level][i].second);
    }
  }

  // Check if we can perform a cheaper "plain copy" (with or without
  // renumbering) instead of having to translate individual entries
  // using copy_indices*. This only works if a) we don't have to send
  // or receive any DoFs and we have all locally owned DoFs in our
  // copy_indices (so no adaptive refinement) and b) all processors
  // agree on the choice (see below).
  const bool my_perform_renumbered_plain_copy =
    (my_copy_indices.back().size() == dof_handler.locally_owned_dofs().n_elements()) &&
    (my_copy_indices_global_mine.back().size() == 0) &&
    (my_copy_indices_level_mine.back().size() == 0);

  bool my_perform_plain_copy = false;
  if(my_perform_renumbered_plain_copy)
  {
    my_perform_plain_copy = true;
    // check whether there is a renumbering of degrees of freedom on
    // either the finest level or the global dofs, which means that we
    // cannot apply a plain copy
    for(dealii::types::global_dof_index i = 0; i < my_copy_indices.back().size(); ++i)
      if(my_copy_indices.back()[i].first != my_copy_indices.back()[i].second)
      {
        my_perform_plain_copy = false;
        break;
      }
  }

  // now do a global reduction over all processors to see what operation
  // they can agree upon
  perform_plain_copy =
    dealii::Utilities::MPI::min(static_cast<int>(my_perform_plain_copy), mpi_communicator);
  perform_renumbered_plain_copy =
    dealii::Utilities::MPI::min(static_cast<int>(my_perform_renumbered_plain_copy),
                                mpi_communicator);

  // if we do a plain copy, no need to hold additional ghosted vectors
  if(perform_renumbered_plain_copy)
  {
  }
}

template class MGTransferH<2, float>;
template class MGTransferH<3, float>;

template class MGTransferH<2, double>;
template class MGTransferH<3, double>;


#define INSTANTIATE_COPY_TO_MG(dim, number_type, vec_number_type)                            \
  template void MGTransferH<dim, number_type>::copy_to_mg(                                   \
    dealii::MGLevelObject<                                                                   \
      dealii::LinearAlgebra::distributed::Vector<number_type, dealii::MemorySpace::CUDA>> &, \
    const dealii::LinearAlgebra::distributed::Vector<vec_number_type,                        \
                                                     dealii::MemorySpace::CUDA> &) const

INSTANTIATE_COPY_TO_MG(2, double, double);
INSTANTIATE_COPY_TO_MG(2, float, float);
INSTANTIATE_COPY_TO_MG(2, float, double);

INSTANTIATE_COPY_TO_MG(3, double, double);
INSTANTIATE_COPY_TO_MG(3, float, float);
INSTANTIATE_COPY_TO_MG(3, float, double);


#define INSTANTIATE_COPY_FROM_MG(dim, number_type, vec_number_type)                           \
  template void MGTransferH<dim, number_type>::copy_from_mg(                                  \
    dealii::LinearAlgebra::distributed::Vector<vec_number_type, dealii::MemorySpace::CUDA> &, \
    const dealii::MGLevelObject<                                                              \
      dealii::LinearAlgebra::distributed::Vector<number_type, dealii::MemorySpace::CUDA>> &) const

INSTANTIATE_COPY_FROM_MG(2, double, double);
INSTANTIATE_COPY_FROM_MG(2, float, float);
INSTANTIATE_COPY_FROM_MG(2, float, double);

INSTANTIATE_COPY_FROM_MG(3, double, double);
INSTANTIATE_COPY_FROM_MG(3, float, float);
INSTANTIATE_COPY_FROM_MG(3, float, double);

} // namespace CUDAWrappers
} // namespace ExaDG
