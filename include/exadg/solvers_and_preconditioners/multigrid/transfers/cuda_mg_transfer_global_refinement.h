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

#ifndef CUDA_MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define CUDA_MG_TRANSFER_MF_MG_LEVEL_OBJECT

// deal.II
#include <deal.II/base/mg_level_object.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

// ExaDG
#include <exadg/matrix_free/cuda_matrix_free.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfers/cuda_mg_transfer.h>

namespace ExaDG
{
namespace CUDAWrappers
{
template<int dim,
         typename Number,
         typename VectorType =
           dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA>>
class MGTransferGlobalRefinement : virtual public MGTransfer<VectorType>
{
public:
  virtual ~MGTransferGlobalRefinement()
  {
  }

  void
  reinit(dealii::Mapping<dim> const & mapping,
         dealii::MGLevelObject<std::shared_ptr<ExaDG::CUDAWrappers::MatrixFree<dim, Number>>> &
                                                                                     mg_matrixfree,
         dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<Number>>> & mg_constraints,
         dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>> & mg_constrained_dofs,
         unsigned int const                                                  dof_handler_index = 0);

  void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const;

  virtual void
  restrict_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const;

  virtual void
  prolongate_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const;

private:
  dealii::MGLevelObject<std::shared_ptr<MGTransfer<VectorType>>> mg_level_object;
};

} // namespace CUDAWrappers
} // namespace ExaDG

#endif
