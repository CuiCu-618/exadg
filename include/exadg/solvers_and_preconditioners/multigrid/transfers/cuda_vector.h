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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_VECTOR_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_VECTOR_H_

#include <deal.II/lac/cuda_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace ExaDG
{
template<typename Number = dealii::types::global_dof_index>
class CudaVector : public dealii::Subscriptor
{
public:
  using value_type      = Number;
  using pointer         = value_type *;
  using const_pointer   = const value_type *;
  using iterator        = value_type *;
  using const_iterator  = const value_type *;
  using reference       = value_type &;
  using const_reference = const value_type &;
  using size_type       = dealii::types::global_dof_index;

  CudaVector();

  CudaVector(const CudaVector<Number> & V);

  ~CudaVector();

  void
  reinit(const size_type size, const bool omit_zeroing_entries = false);

  void
  import(const dealii::LinearAlgebra::ReadWriteVector<Number> & V,
         dealii::VectorOperation::values                        operation);

  Number *
  get_values() const;

  size_type
  size() const;

  std::size_t
  memory_consumption() const;

private:
  std::unique_ptr<Number[], void (*)(Number *)> val;

  size_type n_elements;
};

// ---------------------------- Inline functions --------------------------

template<typename Number>
inline Number *
CudaVector<Number>::get_values() const
{
  return val.get();
}

template<typename Number>
inline typename CudaVector<Number>::size_type
CudaVector<Number>::size() const
{
  return n_elements;
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_CUDA_VECTOR_H_ */
