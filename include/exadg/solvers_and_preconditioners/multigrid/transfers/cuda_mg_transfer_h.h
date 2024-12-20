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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_

// deal.II
#include <deal.II/base/mg_level_object.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
// #include <deal.II/multigrid/mg_transfer_matrix_free.h>

// ExaDG
#include <exadg/operators/cuda_multigrid_operator_base.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/cuda_mg_transfer.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/cuda_vector.h>

namespace ExaDG
{
namespace CUDAWrappers
{
struct IndexMapping
{
  CudaVector<dealii::types::global_dof_index> global_indices;
  CudaVector<dealii::types::global_dof_index> level_indices;
};


/**
 * Specialized matrix-free implementation that overloads the copy_to_mg function for proper
 * initialization of the vectors in matrix-vector products.
 */
template<int dim, typename Number>
class MGTransferH
  : public MGTransfer<dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA>>
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> VectorType;

  MGTransferH(std::map<unsigned int, unsigned int> level_to_triangulation_level_map,
              dealii::DoFHandler<dim> const &      dof_handler);

  virtual ~MGTransferH()
  {
  }

  void
  initialize_constraints(const dealii::MGConstrainedDoFs & mg_constrained_dofs);

  void
  build(const dealii::DoFHandler<dim, dim> & mg_dof,
        const std::vector<std::shared_ptr<const dealii::Utilities::MPI::Partitioner>> &
          external_partitioners =
            std::vector<std::shared_ptr<const dealii::Utilities::MPI::Partitioner>>());


  void
  set_operator(
    dealii::MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> const & operator_in);

  virtual void
  prolongate_and_add(unsigned int const to_level, VectorType & dst, VectorType const & src) const;

  virtual void
  restrict_and_add(unsigned int const from_level, VectorType & dst, VectorType const & src) const;

  virtual void
  interpolate(unsigned int const level_in, VectorType & dst, VectorType const & src) const;

  template<typename VectorType2>
  void
  copy_to_mg(dealii::MGLevelObject<VectorType> & dst, VectorType2 const & src) const;

  template<typename VectorType2>
  void
  copy_from_mg(VectorType2 & dst, dealii::MGLevelObject<VectorType> const & src) const;

private:
  dealii::MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> const *
    underlying_operator;

  /*
   * This map converts the multigrid level as used in the V-cycle to an actual level in the
   * triangulation (this is necessary since both numbers might not equal e.g. in the case of hybrid
   * multigrid involving p-transfer on the same triangulation level).
   */
  mutable std::map<unsigned int, unsigned int> level_to_triangulation_level_map;

  dealii::DoFHandler<dim> const & dof_handler;

  unsigned int fe_degree;

  bool element_is_continuous;

  unsigned int n_components;

  unsigned int n_child_cell_dofs;

  std::vector<CudaVector<dealii::types::global_dof_index>> level_dof_indices;

  std::vector<CudaVector<dealii::types::global_dof_index>> child_offset_in_parent;

  std::vector<unsigned int> n_owned_level_cells;

  VectorType prolongation_matrix_1d;

  std::vector<VectorType> weights_on_refined;

  std::vector<IndexMapping> copy_indices;
  std::vector<IndexMapping> copy_indices_level_mine;

  std::vector<IndexMapping>                   copy_indices_global_mine;
  std::vector<dealii::Table<2, unsigned int>> copy_indices_global_mine_host;

  bool perform_plain_copy;
  bool perform_renumbered_plain_copy;

  std::vector<CudaVector<dealii::types::global_dof_index>> dirichlet_indices;

  dealii::MGLevelObject<std::shared_ptr<const dealii::Utilities::MPI::Partitioner>>
    vector_partitioners;

  dealii::SmartPointer<const dealii::MGConstrainedDoFs, MGTransferH<dim, Number>>
    mg_constrained_dofs;

  mutable VectorType ghosted_global_vector;
  mutable VectorType solution_ghosted_global_vector;

  mutable dealii::MGLevelObject<VectorType> ghosted_level_vector;
  mutable dealii::MGLevelObject<VectorType> solution_ghosted_level_vector;

  void
  fill_copy_indices(const dealii::DoFHandler<dim> & mg_dof);

  template<typename VectorType1, typename VectorType2>
  void
  copy_to_device(VectorType1 & device, const VectorType2 & host);

  template<template<int, int, typename> class loop_body, int degree>
  void
  coarse_cell_loop(const unsigned int fine_level, VectorType & dst, const VectorType & src) const;

  void
  set_mg_constrained_dofs(VectorType & vec, unsigned int level, Number val) const;
};
} // namespace CUDAWrappers
} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_MULTIGRID_TRANSFER_MG_TRANSFER_H_H_ */
