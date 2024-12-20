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

#ifndef CUDA_MG_TRANSFER_MF
#define CUDA_MG_TRANSFER_MF

#include <deal.II/base/mg_level_object.h>

namespace ExaDG
{
namespace CUDAWrappers
{
template<typename VectorType>
class MGTransfer
{
public:
  virtual ~MGTransfer()
  {
  }

  virtual void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;

  virtual void
  restrict_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;

  virtual void
  prolongate_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;

  template<typename VectorType2>
  void
  copy_to_mg(dealii::MGLevelObject<VectorType> & dst, VectorType2 const & src) const;

  template<typename VectorType2>
  void
  copy_from_mg(VectorType2 & dst, dealii::MGLevelObject<VectorType> const & src) const;
};
} // namespace CUDAWrappers
} // namespace ExaDG

#endif
