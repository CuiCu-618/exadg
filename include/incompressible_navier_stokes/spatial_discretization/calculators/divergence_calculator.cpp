/*
 * divergence_calculator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "divergence_calculator.h"

namespace IncNS
{
template<int dim, typename Number>
DivergenceCalculator<dim, Number>::DivergenceCalculator()
  : matrix_free(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::initialize(MatrixFree<dim, Number> const & matrix_free_in,
                                              unsigned int const              dof_index_u_in,
                                              unsigned int const              dof_index_u_scalar_in,
                                              unsigned int const              quad_index_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::compute_divergence(VectorType &       dst,
                                                      VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                             VectorType &                    dst,
                                             VectorType const &              src,
                                             Range const &                   cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, false, true);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      scalar div = integrator_vector.get_divergence(q);
      integrator_scalar.submit_value(div, q);
    }

    integrator_scalar.integrate_scatter(true, false, dst);
  }
}

template class DivergenceCalculator<2, float>;
template class DivergenceCalculator<2, double>;

template class DivergenceCalculator<3, float>;
template class DivergenceCalculator<3, double>;

} // namespace IncNS
