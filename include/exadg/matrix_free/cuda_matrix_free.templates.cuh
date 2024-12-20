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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/matrix_free/shape_info.h>

namespace ExaDG
{
namespace CUDAWrappers
{
template<int dim, typename IteratorFiltersType>
struct IteratorFiltersTypeSelector
{
};
template<int dim>
struct IteratorFiltersTypeSelector<dim, dealii::IteratorFilters::LocallyOwnedCell>
{
  typedef typename dealii::DoFHandler<dim>::active_cell_iterator CellIterator;
  typedef typename dealii::FilteredIterator<CellIterator>        CellFilter;
};
template<int dim>
struct IteratorFiltersTypeSelector<dim, dealii::IteratorFilters::LocallyOwnedLevelCell>
{
  typedef typename dealii::DoFHandler<dim>::level_cell_iterator CellIterator;
  typedef typename dealii::FilteredIterator<CellIterator>       CellFilter;
};

/**
 * Transpose a N x M matrix stored in a one-dimensional array to a M x N
 * matrix stored in a one-dimensional array.
 */
template<typename Number>
void
transpose(const unsigned int N, const unsigned M, const Number * src, Number * dst)
{
  // src is N X M
  // dst is M X N
  for(unsigned int i = 0; i < N; ++i)
    for(unsigned int j = 0; j < M; ++j)
      dst[j * N + i] = src[i * M + j];
}

/**
 * Same as above but the source and the destination are the same vector.
 */
template<typename Number>
void
transpose_in_place(std::vector<Number> & array_host, const unsigned int n, const unsigned int m)
{
  // convert to structure-of-array
  std::vector<Number> old(array_host.size());
  old.swap(array_host);

  transpose(n, m, old.data(), array_host.data());
}

/**
 * Allocate an array to the device and copy @p array_host to the device.
 */
template<typename Number1>
void
alloc_and_copy(Number1 **                                                        array_device,
               const dealii::ArrayView<const Number1, dealii::MemorySpace::Host> array_host,
               const unsigned int                                                n)
{
  cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
  AssertCuda(error_code);
  AssertDimension(array_host.size(), n);

  error_code =
    cudaMemcpy(*array_device, array_host.data(), n * sizeof(Number1), cudaMemcpyHostToDevice);
  AssertCuda(error_code);
}


/**
 * Helper class to (re)initialize MatrixFree object.
 */
template<int dim, typename Number>
class ReinitHelper
{
public:
  ReinitHelper(MatrixFree<dim, Number> *               data,
               const dealii::Mapping<dim> &            mapping,
               const dealii::FiniteElement<dim, dim> & fe,
               const dealii::Quadrature<1> &           quad);

  void
  setup_color_arrays(const unsigned int n_colors);

  void
  setup_cell_arrays(const unsigned int color);

  void
  setup_face_arrays(const unsigned int color);

  template<typename CellFilter>
  void
  get_cell_data(const CellFilter &                                                 cell,
                unsigned int &                                                     cell_id,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner,
                const unsigned int                                                 color);

  template<typename CellFilter>
  void
  get_face_data(CellFilter &                                                       cell,
                unsigned int &                                                     inner_face_id,
                unsigned int &                                                     boundary_face_id,
                const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner,
                const unsigned int                                                 color);

  void
  alloc_and_copy_arrays(const unsigned int color);

  void
  alloc_and_copy_face_arrays(const unsigned int color);

private:
  MatrixFree<dim, Number> * data;
  // Host data
  std::vector<dealii::types::global_dof_index> local_to_global_host;
  std::vector<dealii::types::global_dof_index> face_local_to_global_host;
  std::vector<dealii::types::global_dof_index> cell2face_id_host;
  std::vector<dealii::Point<dim, Number>>      q_points_host;
  std::vector<dealii::Point<dim, Number>>      face_q_points_host;
  std::vector<Number>                          JxW_host;
  std::vector<Number>                          inv_jacobian_host;
  std::vector<Number>                          face_JxW_host;
  std::vector<Number>                          face_inv_jacobian_host;
  std::vector<Number>                          normal_vector_host;
  std::vector<unsigned int>                    face_number_host;
  std::vector<unsigned int>                    boundary_id_host;
  // Local buffer
  std::vector<dealii::types::global_dof_index>          local_dof_indices;
  dealii::FEValues<dim>                                 fe_values;
  dealii::FEFaceValues<dim>                             fe_face_values;
  unsigned int                                          n_inner_faces;
  unsigned int                                          n_boundary_faces;
  const unsigned int                                    fe_degree;
  const unsigned int                                    dofs_per_cell;
  const unsigned int                                    dofs_per_face;
  const unsigned int                                    q_points_per_cell;
  const unsigned int                                    q_points_per_face;
  const unsigned int                                    padding_length;
  const unsigned int                                    face_padding_length;
  const unsigned int                                    mg_level;
  std::vector<std::map<std::tuple<int, int, int>, int>> cell2cell_id;
};

template<int dim, typename Number>
ReinitHelper<dim, Number>::ReinitHelper(MatrixFree<dim, Number> *          data,
                                        const dealii::Mapping<dim> &       mapping,
                                        const dealii::FiniteElement<dim> & fe,
                                        const dealii::Quadrature<1> &      quad)
  : data(data),
    fe_degree(data->fe_degree),
    dofs_per_cell(data->dofs_per_cell),
    dofs_per_face(data->dofs_per_face),
    q_points_per_cell(data->q_points_per_cell),
    q_points_per_face(data->q_points_per_face),
    fe_values(mapping,
              fe,
              dealii::Quadrature<dim>(quad),
              dealii::update_inverse_jacobians | dealii::update_quadrature_points |
                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values),
    fe_face_values(mapping,
                   fe,
                   dealii::Quadrature<dim - 1>(quad),
                   dealii::update_inverse_jacobians | dealii::update_quadrature_points |
                     dealii::update_normal_vectors | dealii::update_values |
                     dealii::update_gradients | dealii::update_JxW_values),
    padding_length(data->get_padding_length()),
    face_padding_length(data->get_face_padding_length()),
    mg_level(data->mg_level),
    n_inner_faces(0),
    n_boundary_faces(0)
{
  local_dof_indices.resize(data->dofs_per_cell);
}

template<int dim, typename Number>
void
ReinitHelper<dim, Number>::setup_color_arrays(const unsigned int n_colors)
{
  data->n_cells.resize(n_colors, 0);
  data->n_inner_faces.resize(n_colors, 0);
  data->n_boundary_faces.resize(n_colors, 0);
  data->grid_dim.resize(n_colors);
  data->grid_dim_inner_face.resize(n_colors);
  data->grid_dim_boundary_face.resize(n_colors);
  data->block_dim.resize(n_colors);
  data->block_dim_inner_face.resize(n_colors);
  data->block_dim_boundary_face.resize(n_colors);
  data->local_to_global.resize(n_colors);
  data->face_local_to_global.resize(n_colors);
  data->cell2face_id.resize(n_colors);
  data->face_number.resize(n_colors);
  data->boundary_id.resize(n_colors);
  cell2cell_id.resize(n_colors);

  data->row_start.resize(n_colors);

  data->q_points.resize(n_colors);
  data->JxW.resize(n_colors);
  data->inv_jacobian.resize(n_colors);

  data->face_q_points.resize(n_colors);
  data->face_JxW.resize(n_colors);
  data->face_inv_jacobian.resize(n_colors);
  data->normal_vector.resize(n_colors);
}

template<int dim, typename Number>
void
ReinitHelper<dim, Number>::setup_cell_arrays(const unsigned int color)
{
  const unsigned int n_cells         = data->n_cells[color];
  const unsigned int cells_per_block = data->cells_per_block;

  // Setup kernel parameters
  double apply_n_blocks =
    std::ceil(static_cast<double>(n_cells) / static_cast<double>(cells_per_block));
  data->grid_dim[color] = dim3(apply_n_blocks);

  const unsigned int n_dofs_1d = fe_degree + 1;
  data->block_dim[color]       = dim3(n_dofs_1d, n_dofs_1d * cells_per_block);

  local_to_global_host.resize(n_cells);

  q_points_host.resize(n_cells * padding_length);
  JxW_host.resize(n_cells * padding_length);
  inv_jacobian_host.resize(n_cells * padding_length * dim * dim);
}

template<int dim, typename Number>
void
ReinitHelper<dim, Number>::setup_face_arrays(const unsigned int color)
{
  data->n_inner_faces[color]               = n_inner_faces;
  const unsigned int inner_faces_per_block = data->inner_faces_per_block;

  data->n_boundary_faces[color]               = n_boundary_faces;
  const unsigned int boundary_faces_per_block = data->boundary_faces_per_block;

  const unsigned int n_faces = n_inner_faces * 2 + n_boundary_faces;

  // Setup kernel parameters
  double apply_n_blocks =
    std::ceil(static_cast<double>(n_inner_faces) / static_cast<double>(inner_faces_per_block));
  data->grid_dim_inner_face[color] = dim3(apply_n_blocks);

  apply_n_blocks                      = std::ceil(static_cast<double>(n_boundary_faces) /
                             static_cast<double>(boundary_faces_per_block));
  data->grid_dim_boundary_face[color] = dim3(apply_n_blocks);

  // TODO this should be a templated parameter.
  const unsigned int n_dofs_1d         = fe_degree + 1;
  data->block_dim_inner_face[color]    = dim3(n_dofs_1d, n_dofs_1d * inner_faces_per_block);
  data->block_dim_boundary_face[color] = dim3(n_dofs_1d, n_dofs_1d * boundary_faces_per_block);

  face_local_to_global_host.resize(n_faces);
  cell2face_id_host.resize(data->n_cells[color] * (dim * 2) * 2);
  face_number_host.resize(n_faces);
  boundary_id_host.resize(n_boundary_faces);

  face_q_points_host.resize(n_faces * face_padding_length);
  face_JxW_host.resize(n_faces * face_padding_length);
  face_inv_jacobian_host.resize(n_faces * face_padding_length * dim * dim);
  normal_vector_host.resize(n_faces * face_padding_length * dim * 1);

  // // Debug output
  // {
  //   std::cout << color << ": " << n_inner_faces << " " << n_boundary_faces << std::endl;
  // }
}

template<int dim, typename Number>
template<typename CellFilter>
void
ReinitHelper<dim, Number>::get_cell_data(
  const CellFilter &                                                 cell,
  unsigned int &                                                     cell_id,
  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner,
  const unsigned int                                                 color)
{
  auto fill_index_data = [&](auto & c, auto obj_id)
  {
    c->get_active_or_mg_dof_indices(local_dof_indices);

    // When using MPI, we need to transform the local_dof_indices, which
    // contains global dof indices, to get local (to the current MPI
    // process) dof indices.
    if(partitioner)
      local_dof_indices[0] = partitioner->global_to_local(local_dof_indices[0]);

    local_to_global_host[obj_id] = local_dof_indices[0];
  };

  auto fill_cell_data = [&](auto & fe_value, auto obj_id)
  {
    // Quadrature points
    {
      const std::vector<dealii::Point<dim>> & q_points = fe_value.get_quadrature_points();
      std::copy(q_points.begin(), q_points.end(), &q_points_host[obj_id * padding_length]);
    }
    // JxW
    {
      std::vector<double> JxW_values_double = fe_value.get_JxW_values();
      const unsigned int  offset            = obj_id * padding_length;
      for(unsigned int i = 0; i < q_points_per_cell; ++i)
        JxW_host[i + offset] = static_cast<Number>(JxW_values_double[i]);
    }
    // Inverse Jacobians
    {
      const std::vector<dealii::DerivativeForm<1, dim, dim>> & inv_jacobians =
        fe_value.get_inverse_jacobians();
      std::copy(&inv_jacobians[0][0][0],
                &inv_jacobians[0][0][0] +
                  q_points_per_cell * sizeof(dealii::DerivativeForm<1, dim, dim>) / sizeof(double),
                &inv_jacobian_host[obj_id * padding_length * dim * dim]);
    }
  };


  {
    auto cell_info =
      std::make_tuple<int, int, int>(cell->level_subdomain_id(), cell->level(), cell->index());
    cell2cell_id[color][cell_info] = cell_id;

    fill_index_data(cell, cell_id);

    fe_values.reinit(cell);
    fill_cell_data(fe_values, cell_id);

    cell_id++;
  }

  // // Debug output
  // {
  //   std::cout << cell_id << " " << cell << std::endl;
  // }

  for(const unsigned int face_no : cell->face_indices())
  {
    if(cell->at_boundary(face_no))
    {
      n_boundary_faces++;
    }
    else
    {
      auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);

      if(cell->neighbor_is_coarser(face_no))
      {
        AssertThrow(false, dealii::ExcMessage("Local refinement not implemented. TODO"));
      }
      else
      {
        if(neighbor < cell)
          continue;

        n_inner_faces++;
      }
    }
  }
}

template<int dim, typename Number>
template<typename CellFilter>
void
ReinitHelper<dim, Number>::get_face_data(
  CellFilter &                                                       cell,
  unsigned int &                                                     inner_face_id,
  unsigned int &                                                     boundary_face_id,
  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner,
  const unsigned int                                                 color)
{
  auto fill_data = [&](auto & fe_value, auto obj_id)
  {
    {
      const std::vector<dealii::Point<dim>> & q_points = fe_value.get_quadrature_points();
      std::copy(q_points.begin(),
                q_points.end(),
                &face_q_points_host[obj_id * face_padding_length]);

      // std::cout << "obj_id: " << obj_id << " --- ";
      // for(auto q : q_points)
      //   std::cout << q << " ";
      // std::cout << std::endl;
    }

    {
      std::vector<double> JxW_values = fe_value.get_JxW_values();
      const unsigned int  offset     = obj_id * face_padding_length;
      for(unsigned int i = 0; i < q_points_per_face; ++i)
        face_JxW_host[i + offset] = static_cast<Number>(JxW_values[i]);
    }

    {
      const std::vector<dealii::DerivativeForm<1, dim, dim>> & inv_jacobians =
        fe_value.get_inverse_jacobians();
      std::copy(&inv_jacobians[0][0][0],
                &inv_jacobians[0][0][0] +
                  q_points_per_face * sizeof(dealii::DerivativeForm<1, dim, dim>) / sizeof(double),
                &face_inv_jacobian_host[obj_id * face_padding_length * dim * dim]);
    }

    {
      const std::vector<dealii::Tensor<1, dim>> & normal_vectors = fe_value.get_normal_vectors();
      std::copy(&normal_vectors[0][0],
                &normal_vectors[0][0] +
                  q_points_per_face * sizeof(dealii::Tensor<1, dim>) / sizeof(double),
                &normal_vector_host[obj_id * face_padding_length * dim * 1]);
    }
  };

  auto get_first_dof_index = [&](auto & c)
  {
    c->get_active_or_mg_dof_indices(local_dof_indices);

    if(partitioner)
      local_dof_indices[0] = partitioner->global_to_local(local_dof_indices[0]);

    return local_dof_indices[0];
  };

  for(const unsigned int face_no : cell->face_indices())
  {
    auto cell_info =
      std::make_tuple<int, int, int>(cell->level_subdomain_id(), cell->level(), cell->index());
    auto cell_id = cell2cell_id[color][cell_info];

    if(cell->at_boundary(face_no))
    {
      face_local_to_global_host[n_inner_faces * 2 + boundary_face_id] = get_first_dof_index(cell);
      face_number_host[n_inner_faces * 2 + boundary_face_id]          = face_no;

      fe_face_values.reinit(cell, face_no);
      fill_data(fe_face_values, n_inner_faces * 2 + boundary_face_id);

      cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2] =
        n_inner_faces * 2 + boundary_face_id;
      cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2 + 1] =
        n_inner_faces * 2 + boundary_face_id;

      boundary_id_host[boundary_face_id] = cell->face(face_no)->boundary_id();

      // std::cout << cell << " " << face_no << " " << cell->face(face_no)->boundary_id() << "\n";

      boundary_face_id++;
    }
    else
    {
      auto neighbor = cell->neighbor_or_periodic_neighbor(face_no);

      if(cell->neighbor_is_coarser(face_no))
      {
        AssertThrow(false, dealii::ExcMessage("Local refinement not implemented. TODO"));
      }
      else
      {
        auto neighbor_face_no = cell->neighbor_face_no(face_no);

        if(neighbor < cell)
        {
          // faces handled by other cells
          {
            cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2]     = 0;
            cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2 + 1] = 0;
          }
          continue;
        }

        // // Debug output
        // {
        //   if(neighbor->is_ghost())
        //   {
        //     std::cout << "ghost: " << cell << " " << neighbor << " " << std::endl;
        //   }
        // }

        face_local_to_global_host[2 * inner_face_id] = get_first_dof_index(cell);
        face_number_host[2 * inner_face_id]          = face_no;

        face_local_to_global_host[2 * inner_face_id + 1] = get_first_dof_index(neighbor);
        face_number_host[2 * inner_face_id + 1]          = neighbor_face_no;

        fe_face_values.reinit(cell, face_no);
        fill_data(fe_face_values, 2 * inner_face_id);

        fe_face_values.reinit(neighbor, neighbor_face_no);
        fill_data(fe_face_values, 2 * inner_face_id + 1);

        cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2]     = 2 * inner_face_id;
        cell2face_id_host[cell_id * (dim * 2) * 2 + face_no * 2 + 1] = 2 * inner_face_id + 1;

        inner_face_id++;
      }
    }
  }
}

template<int dim, typename Number>
void
ReinitHelper<dim, Number>::alloc_and_copy_arrays(const unsigned int color)
{
  unsigned int n_cells = data->n_cells[color];

  // Local-to-global mapping
  alloc_and_copy(&data->local_to_global[color],
                 dealii::ArrayView<const dealii::types::global_dof_index>(
                   local_to_global_host.data(), local_to_global_host.size()),
                 n_cells);

  // Quadrature points
  {
    alloc_and_copy(&data->q_points[color],
                   dealii::ArrayView<const dealii::Point<dim, Number>>(q_points_host.data(),
                                                                       q_points_host.size()),
                   n_cells * padding_length);
  }

  // Jacobian determinants/quadrature weights
  {
    alloc_and_copy(&data->JxW[color],
                   dealii::ArrayView<const Number>(JxW_host.data(), JxW_host.size()),
                   n_cells * padding_length);
  }

  // Inverse jacobians
  {
    // Reorder so that all J_11 elements are together, all J_12 elements
    // are together, etc., i.e., reorder indices from
    // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
    // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
    transpose_in_place(inv_jacobian_host, padding_length * n_cells, dim * dim);

    alloc_and_copy(&data->inv_jacobian[color],
                   dealii::ArrayView<const Number>(inv_jacobian_host.data(),
                                                   inv_jacobian_host.size()),
                   n_cells * dim * dim * padding_length);
  }
}

template<int dim, typename Number>
void
ReinitHelper<dim, Number>::alloc_and_copy_face_arrays(const unsigned int color)
{
  const unsigned int n_faces = data->n_inner_faces[color] * 2 + data->n_boundary_faces[color];

  alloc_and_copy(&data->face_number[color],
                 dealii::ArrayView<const unsigned int>(face_number_host.data(),
                                                       face_number_host.size()),
                 n_faces);

  alloc_and_copy(&data->boundary_id[color],
                 dealii::ArrayView<const unsigned int>(boundary_id_host.data(),
                                                       boundary_id_host.size()),
                 data->n_boundary_faces[color]);

  alloc_and_copy(&data->face_local_to_global[color],
                 dealii::ArrayView<const dealii::types::global_dof_index>(
                   face_local_to_global_host.data(), face_local_to_global_host.size()),
                 n_faces);

  alloc_and_copy(&data->cell2face_id[color],
                 dealii::ArrayView<const dealii::types::global_dof_index>(cell2face_id_host.data(),
                                                                          cell2face_id_host.size()),
                 data->n_cells[color] * (dim * 2) * 2);

  // Quadrature points
  {
    alloc_and_copy(&data->face_q_points[color],
                   dealii::ArrayView<const dealii::Point<dim, Number>>(face_q_points_host.data(),
                                                                       face_q_points_host.size()),
                   n_faces * face_padding_length);
  }

  // Face jacobian determinants/quadrature weights
  {
    alloc_and_copy(&data->face_JxW[color],
                   dealii::ArrayView<const Number>(face_JxW_host.data(), face_JxW_host.size()),
                   n_faces * face_padding_length);
  }

  // face Inverse jacobians
  {
    // Reorder so that all J_11 elements are together, all J_12 elements
    // are together, etc., i.e., reorder indices from
    // cell_id*q_points_per_cell*dim*dim + q*dim*dim +i to
    // i*q_points_per_cell*n_cells + cell_id*q_points_per_cell+q
    transpose_in_place(face_inv_jacobian_host, n_faces * face_padding_length, dim * dim);

    alloc_and_copy(&data->face_inv_jacobian[color],
                   dealii::ArrayView<const Number>(face_inv_jacobian_host.data(),
                                                   face_inv_jacobian_host.size()),
                   n_faces * dim * dim * face_padding_length);
  }

  // face normal vectors
  {
    transpose_in_place(normal_vector_host, n_faces * face_padding_length, dim * 1);

    alloc_and_copy(&data->normal_vector[color],
                   dealii::ArrayView<const Number>(normal_vector_host.data(),
                                                   normal_vector_host.size()),
                   n_faces * dim * 1 * face_padding_length);
  }

  n_inner_faces    = 0;
  n_boundary_faces = 0;
}

namespace internal
{
extern __shared__ double data_dd[];
extern __shared__ float  data_ff[];

template<typename Number>
__device__ inline Number *
get_shared_data();

template<>
__device__ inline double *
get_shared_data()
{
  return data_dd;
}

template<>
__device__ inline float *
get_shared_data()
{
  return data_ff;
}
} // namespace internal

template<int dim, typename Number, typename Functor>
__global__ void __launch_bounds__(256, 1)
  apply_kernel_shmem(Functor                                func,
                     typename MatrixFree<dim, Number>::Data gpu_data,
                     const Number *                         src,
                     Number *                               dst)
{
  constexpr unsigned int n_dofs_2d        = Functor::n_dofs_1d * Functor::n_dofs_1d;
  constexpr unsigned int cells_per_block  = Functor::cells_per_block;
  constexpr unsigned int n_dofs_per_block = cells_per_block * Functor::n_local_dofs;

  const unsigned int local_cell = threadIdx.y / Functor::n_dofs_1d;
  const unsigned int cell       = local_cell + cells_per_block * blockIdx.x;

  SharedData<dim, Number> shared_data(internal::get_shared_data<Number>(),
                                      local_cell,
                                      n_dofs_2d,
                                      Functor::n_local_dofs,
                                      cells_per_block);

  if(cell < gpu_data.n_objs)
    func(cell, &gpu_data, &shared_data, src, dst);
}

template<int dim, typename Number, typename Functor>
__global__ void
evaluate_coeff(Functor func, const typename MatrixFree<dim, Number>::Data gpu_data)
{
  const unsigned int local_cell = threadIdx.y / Functor::n_dofs_1d;
  const unsigned int cell       = local_cell + Functor::cells_per_block * blockIdx.x;

  if(cell < gpu_data.n_objs)
    func(cell, &gpu_data);
}

template<int dim, typename Number, typename Functor>
void
allocate_shared_memory()
{
  AssertCuda(cudaFuncSetAttribute(apply_kernel_shmem<dim, Number, Functor>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  Functor::shared_mem));
}

template<int dim, typename Number>
MatrixFree<dim, Number>::MatrixFree()
  : dealii::Subscriptor(), mg_level(dealii::numbers::invalid_unsigned_int)
{
}

template<int dim, typename Number>
MatrixFree<dim, Number>::~MatrixFree()
{
  free();
}

template<int dim, typename Number>
template<typename IteratorFiltersType>
void
MatrixFree<dim, Number>::reinit(
  const dealii::Mapping<dim> &                                     mapping,
  const dealii::DoFHandler<dim> &                                  dof_handler,
  const dealii::AffineConstraints<Number> &                        constraints,
  const dealii::Quadrature<dim> &                                  quad,
  const IteratorFiltersType &                                      iterator_filter,
  const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data)
{
  const auto & triangulation = dof_handler.get_triangulation();
  if(const auto parallel_triangulation =
       dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(&triangulation))
    internal_reinit(mapping,
                    dof_handler,
                    constraints,
                    quad,
                    iterator_filter,
                    std::make_shared<const MPI_Comm>(parallel_triangulation->get_communicator()),
                    additional_data);
  else
    internal_reinit(
      mapping, dof_handler, constraints, quad, iterator_filter, nullptr, additional_data);
}

template<int dim, typename Number>
void
MatrixFree<dim, Number>::reinit(
  const dealii::Mapping<dim> &                                     mapping,
  const dealii::DoFHandler<dim> &                                  dof_handler,
  const dealii::AffineConstraints<Number> &                        constraints,
  const dealii::Quadrature<dim> &                                  quad,
  const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data)
{
  dealii::IteratorFilters::LocallyOwnedCell locally_owned_cell_filter;
  reinit(mapping, dof_handler, constraints, quad, locally_owned_cell_filter, additional_data);
}

template<int dim, typename Number>
template<typename IteratorFiltersType>
void
MatrixFree<dim, Number>::internal_reinit(
  const dealii::Mapping<dim> &                                     mapping,
  const dealii::DoFHandler<dim> &                                  dof_handler_,
  const dealii::AffineConstraints<Number> &                        constraints,
  const dealii::Quadrature<dim> &                                  quad,
  const IteratorFiltersType &                                      iterator_filter,
  std::shared_ptr<const MPI_Comm>                                  comm,
  const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data)
{
  if(typeid(Number) == typeid(double))
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  this->dof_handler                       = &dof_handler_;
  this->overlap_communication_computation = additional_data.overlap_communication_computation;
  this->mg_level                          = additional_data.mg_level;

  // this->overlap_communication_computation = false;

  this->geometry_type = dealii::internal::MatrixFreeFunctions::GeometryType::general;

  n_dofs = mg_level == dealii::numbers::invalid_unsigned_int ? dof_handler->n_dofs() :
                                                               dof_handler->n_dofs(mg_level);

  const dealii::FiniteElement<dim> & fe = dof_handler->get_fe();

  fe_degree                    = fe.degree;
  const unsigned int n_dofs_1d = fe_degree + 1;

  dealii::QGauss<1>  quad_1d(n_dofs_1d);
  const unsigned int n_q_points_1d = quad_1d.size();

  Assert(n_dofs_1d == n_q_points_1d && std::pow(n_q_points_1d, dim) == quad.size(),
         dealii::ExcMessage("So far, n_q_points_1d must be equal to fe_degree + 1."));

  // Set padding length to the closest power of two larger than or equal to
  // the number of threads.
  // TODO: check the performance number without padding.
  padding_length = 1 << static_cast<unsigned int>(std::ceil(dim * std::log2(fe_degree + 1.)));

  face_padding_length =
    1 << static_cast<unsigned int>(std::ceil((dim - 1) * std::log2(fe_degree + 1.)));

  dofs_per_cell     = fe.n_dofs_per_cell();
  q_points_per_cell = std::pow(n_q_points_1d, dim);
  q_points_per_face = std::pow(n_q_points_1d, dim - 1);

  const unsigned int n_shape_values = n_dofs_1d * n_q_points_1d;

  const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(quad_1d, fe);

  dealii::FE_DGQArbitraryNodes<1>                                fe_quad_co(quad_1d);
  const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info_co(quad_1d, fe_quad_co);

  // copy shape info to device
  {
    std::vector<Number> cell_face_shape_value_host;
    cell_face_shape_value_host.resize(3 * n_shape_values);

    std::vector<Number> cell_face_shape_gradients_host;
    cell_face_shape_gradients_host.resize(3 * n_shape_values);

    std::vector<Number> cell_face_co_shape_gradients_host;
    cell_face_co_shape_gradients_host.resize(5 * n_shape_values); // D^co, S^co_f, D^co_f

    for(unsigned int i = 0; i < n_shape_values; ++i)
    {
      cell_face_shape_value_host[i]        = shape_info.data.front().shape_values[i];
      cell_face_shape_gradients_host[i]    = shape_info.data.front().shape_gradients[i];
      cell_face_co_shape_gradients_host[i] = shape_info_co.data.front().shape_gradients[i];
    }

    for(unsigned int i = 0; i < n_q_points_1d; ++i)
    {
      cell_face_shape_value_host[n_shape_values + i * n_q_points_1d] =
        shape_info.data.front().shape_data_on_face[0][i];
      cell_face_shape_value_host[2 * n_shape_values + i * n_q_points_1d] =
        shape_info.data.front().shape_data_on_face[1][i];

      cell_face_shape_gradients_host[n_shape_values + i * n_q_points_1d] =
        shape_info.data.front().shape_data_on_face[0][n_q_points_1d + i];
      cell_face_shape_gradients_host[2 * n_shape_values + i * n_q_points_1d] =
        shape_info.data.front().shape_data_on_face[1][n_q_points_1d + i];

      cell_face_co_shape_gradients_host[n_shape_values + i * n_q_points_1d] =
        shape_info_co.data.front().shape_data_on_face[0][i];
      cell_face_co_shape_gradients_host[2 * n_shape_values + i * n_q_points_1d] =
        shape_info_co.data.front().shape_data_on_face[1][i];

      cell_face_co_shape_gradients_host[3 * n_shape_values + i * n_q_points_1d] =
        shape_info_co.data.front().shape_data_on_face[0][n_q_points_1d + i];
      cell_face_co_shape_gradients_host[4 * n_shape_values + i * n_q_points_1d] =
        shape_info_co.data.front().shape_data_on_face[1][n_q_points_1d + i];
    }

    alloc_and_copy(&cell_face_shape_values,
                   dealii::ArrayView<const Number>(cell_face_shape_value_host.data(),
                                                   cell_face_shape_value_host.size()),
                   3 * n_shape_values);

    alloc_and_copy(&cell_face_shape_gradients,
                   dealii::ArrayView<const Number>(cell_face_shape_gradients_host.data(),
                                                   cell_face_shape_gradients_host.size()),
                   3 * n_shape_values);

    alloc_and_copy(&cell_face_co_shape_gradients,
                   dealii::ArrayView<const Number>(cell_face_co_shape_gradients_host.data(),
                                                   cell_face_co_shape_gradients_host.size()),
                   5 * n_shape_values);
  }

  // Setup the number of cells per CUDA thread block
  cells_per_block          = cells_per_block_shmem(dim, fe_degree);
  inner_faces_per_block    = faces_per_block_shmem(dim, fe_degree); // TODO:
  boundary_faces_per_block = faces_per_block_shmem(dim, fe_degree); // TODO:

  // create graph coloring
  typedef typename IteratorFiltersTypeSelector<dim, IteratorFiltersType>::CellIterator CellIterator;
  typedef typename IteratorFiltersTypeSelector<dim, IteratorFiltersType>::CellFilter   CellFilter;

  CellIterator beginc;
  CellIterator endc;

  if(mg_level == dealii::numbers::invalid_unsigned_int)
  {
    beginc = dof_handler->begin_active();
    endc   = dof_handler->end();
  }
  else
  {
    AssertIndexRange(mg_level, dof_handler->get_triangulation().n_levels());

    beginc = dof_handler->begin_mg(mg_level);
    endc   = dof_handler->end_mg(mg_level);
  }

  CellFilter begin(iterator_filter, beginc);
  CellFilter end(iterator_filter, endc);

  std::vector<std::vector<CellFilter>> graph;
  if(begin != end)
  {
    graph.clear();
    if(overlap_communication_computation)
    {
      // We create one color (1) with the cells on the boundary of the
      // local domain and two colors (0 and 2) with the interior
      // cells.
      graph.resize(3, std::vector<CellFilter>());

      std::vector<bool> ghost_vertices(dof_handler->get_triangulation().n_vertices(), false);

      for(auto cell = beginc; cell != endc; ++cell)
        if((mg_level == dealii::numbers::invalid_unsigned_int && cell->is_ghost()) ||
           (mg_level != dealii::numbers::invalid_unsigned_int && cell->is_ghost_on_level()))
          for(unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; i++)
            ghost_vertices[cell->vertex_index(i)] = true;

      std::vector<CellFilter> inner_cells;

      for(auto cell = begin; cell != end; ++cell)
      {
        bool ghost_vertex = false;

        for(unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; i++)
          if(ghost_vertices[cell->vertex_index(i)])
          {
            ghost_vertex = true;
            break;
          }

        if(ghost_vertex)
          graph[1].emplace_back(cell);
        else
          inner_cells.emplace_back(cell);
      }

      for(unsigned i = 0; i < inner_cells.size(); ++i)
        if(i < inner_cells.size() / 2)
          graph[0].emplace_back(inner_cells[i]);
        else
          graph[2].emplace_back(inner_cells[i]);
    }
    else
    {
      // If we are not using coloring, all the cells belong to the
      // same color.
      graph.resize(1, std::vector<CellFilter>());
      for(auto cell = begin; cell != end; ++cell)
        graph[0].emplace_back(cell);
    }
  }

  n_colors = graph.size();

  // // Debug output
  // {
  //   std::cout << n_colors << std::endl;
  //   for(unsigned int i = 0; i < n_colors; ++i)
  //   {
  //     for(auto & cell : graph[i])
  //       std::cout << cell << " ";
  //     std::cout << std::endl;
  //   }
  // }

  if(comm)
  {
    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    locally_relevant_dofs =
      mg_level == dealii::numbers::invalid_unsigned_int ?
        dealii::DoFTools::extract_locally_relevant_dofs(*dof_handler) :
        dealii::DoFTools::extract_locally_relevant_level_dofs(*dof_handler, mg_level);
    locally_owned_dofs = mg_level == dealii::numbers::invalid_unsigned_int ?
                           dof_handler->locally_owned_dofs() :
                           dof_handler->locally_owned_mg_dofs(mg_level);

    partitioner = std::make_shared<dealii::Utilities::MPI::Partitioner>(locally_owned_dofs,
                                                                        locally_relevant_dofs,
                                                                        *comm);
  }

  ReinitHelper<dim, Number> helper(this, mapping, fe, quad_1d);
  helper.setup_color_arrays(n_colors);

  for(unsigned int i = 0; i < n_colors; ++i)
  {
    n_cells[i] = graph[i].size();
    helper.setup_cell_arrays(i);

    unsigned int cell_id = 0;
    for(auto cell = graph[i].begin(); cell != graph[i].end(); ++cell)
      helper.get_cell_data(*cell, cell_id, partitioner, i);

    helper.alloc_and_copy_arrays(i);

    helper.setup_face_arrays(i);

    unsigned int inner_face_id    = 0;
    unsigned int boundary_face_id = 0;
    for(auto cell = graph[i].begin(); cell != graph[i].end(); ++cell)
      helper.get_face_data(*cell, inner_face_id, boundary_face_id, partitioner, i);

    helper.alloc_and_copy_face_arrays(i);
  }

  // TODO: CG
  (void)constraints;
}

template<int dim, typename Number>
unsigned int
MatrixFree<dim, Number>::get_mg_level() const
{
  return mg_level;
}

template<int dim, typename Number>
unsigned int
MatrixFree<dim, Number>::get_fe_degree() const
{
  return fe_degree;
}

template<int dim, typename Number>
unsigned int
MatrixFree<dim, Number>::get_padding_length() const
{
  return padding_length;
}

template<int dim, typename Number>
unsigned int
MatrixFree<dim, Number>::get_face_padding_length() const
{
  return face_padding_length;
}

template<int dim, typename Number>
MatrixFree<dim, Number>::Data
MatrixFree<dim, Number>::get_cell_face_data(unsigned int color) const
{
  Data data_copy;

  data_copy.q_points     = q_points[color];
  data_copy.inv_jacobian = inv_jacobian[color];
  data_copy.JxW          = JxW[color];

  data_copy.local_to_global = local_to_global[color];
  data_copy.n_cells         = n_cells[color];
  data_copy.padding_length  = padding_length;
  data_copy.row_start       = row_start[color];
  data_copy.n_objs          = n_cells[color];

  data_copy.cell2face_id = cell2face_id[color];

  data_copy.face_q_points     = face_q_points[color];
  data_copy.face_inv_jacobian = face_inv_jacobian[color];
  data_copy.face_JxW          = face_JxW[color];
  data_copy.normal_vector     = normal_vector[color];

  data_copy.face_local_to_global = face_local_to_global[color];
  data_copy.n_faces              = n_boundary_faces[color] + n_inner_faces[color] * 2;
  data_copy.n_inner_faces        = n_inner_faces[color] * 2;

  data_copy.boundary_id         = boundary_id[color];
  data_copy.face_number         = face_number[color];
  data_copy.face_padding_length = face_padding_length;

  data_copy.cell_face_shape_values       = cell_face_shape_values;
  data_copy.cell_face_shape_gradients    = cell_face_shape_gradients;
  data_copy.cell_face_co_shape_gradients = cell_face_co_shape_gradients;

  return data_copy;
}

template<int dim, typename Number>
MatrixFree<dim, Number>::Data
MatrixFree<dim, Number>::get_boundary_face_data(unsigned int color) const
{
  Data data_copy;

  data_copy.face_q_points = face_q_points[color] + 2 * n_inner_faces[color] * face_padding_length;
  data_copy.face_inv_jacobian =
    face_inv_jacobian[color] + 2 * n_inner_faces[color] * face_padding_length * dim * dim;
  data_copy.face_JxW = face_JxW[color] + 2 * n_inner_faces[color] * face_padding_length;
  data_copy.normal_vector =
    normal_vector[color] + 2 * n_inner_faces[color] * face_padding_length * dim;

  data_copy.face_local_to_global = face_local_to_global[color] + n_inner_faces[color] * 2;
  data_copy.face_number          = face_number[color] + n_inner_faces[color] * 2;
  data_copy.boundary_id          = boundary_id[color];
  data_copy.face_padding_length  = face_padding_length;

  data_copy.n_faces = n_boundary_faces[color] + n_inner_faces[color] * 2;
  data_copy.n_objs  = n_boundary_faces[color];

  data_copy.cell_face_shape_values       = cell_face_shape_values;
  data_copy.cell_face_shape_gradients    = cell_face_shape_gradients;
  data_copy.cell_face_co_shape_gradients = cell_face_co_shape_gradients;

  return data_copy;
}

template<int dim, typename Number>
template<typename Functor, typename VectorType>
void
MatrixFree<dim, Number>::cell_loop(const Functor &    func,
                                   const VectorType & src,
                                   VectorType &       dst) const
{
  if(partitioner)
    distributed_cell_loop(func, src, dst);
  else
    serial_cell_loop(func, src, dst);
}

template<int dim, typename Number>
template<typename Functor, typename VectorType>
void
MatrixFree<dim, Number>::boundary_face_loop(const Functor &    func,
                                            const VectorType & src,
                                            VectorType &       dst) const
{
  allocate_shared_memory<dim, Number, Functor>();

  // Execute the loop on the boundary faces
  for(unsigned int i = 0; i < n_colors; ++i)
    if(n_boundary_faces[i] > 0)
    {
      apply_kernel_shmem<dim, Number, Functor>
        <<<grid_dim_boundary_face[i], block_dim_boundary_face[i], Functor::shared_mem>>>(
          func, get_boundary_face_data(i), src.get_values(), dst.get_values());
      AssertCudaKernel();
    }
}

template<int dim, typename Number>
template<typename Functor>
void
MatrixFree<dim, Number>::evaluate_coefficients(Functor func) const
{
  for(unsigned int i = 0; i < n_colors; ++i)
    if(n_cells[i] > 0)
    {
      evaluate_coeff<dim, Number, Functor>
        <<<grid_dim[i], block_dim[i]>>>(func, get_cell_face_data(i));
      AssertCudaKernel();
    }
}

template<int dim, typename Number>
void
MatrixFree<dim, Number>::initialize_dof_vector(
  dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> & vec,
  const unsigned int) const
{
  if(partitioner)
    vec.reinit(partitioner);
  else
    vec.reinit(n_dofs);
}

template<int dim, typename Number>
void
MatrixFree<dim, Number>::free()
{
  auto free_device_data = [](auto device_data)
  {
    for(auto & color : device_data)
      dealii::Utilities::CUDA::free(color);
    device_data.clear();
  };

  free_device_data(q_points);
  free_device_data(face_q_points);
  free_device_data(local_to_global);
  free_device_data(cell2face_id);
  free_device_data(face_local_to_global);

  free_device_data(inv_jacobian);
  free_device_data(JxW);
  free_device_data(face_inv_jacobian);
  free_device_data(face_JxW);
  free_device_data(normal_vector);
  free_device_data(face_number);
  free_device_data(boundary_id);
}

template<int dim, typename Number>
template<typename Functor, typename VectorType>
void
MatrixFree<dim, Number>::serial_cell_loop(const Functor &    func,
                                          const VectorType & src,
                                          VectorType &       dst) const
{
  allocate_shared_memory<dim, Number, Functor>();

  // Execute the loop on the cells
  for(unsigned int i = 0; i < n_colors; ++i)
    if(n_cells[i] > 0)
    {
      apply_kernel_shmem<dim, Number, Functor><<<grid_dim[i], block_dim[i], Functor::shared_mem>>>(
        func, get_cell_face_data(i), src.get_values(), dst.get_values());
      AssertCudaKernel();
    }
}

template<int dim, typename Number>
template<typename Functor, typename VectorType>
void
MatrixFree<dim, Number>::distributed_cell_loop(const Functor &    func,
                                               const VectorType & src,
                                               VectorType &       dst) const
{
  allocate_shared_memory<dim, Number, Functor>();

  // in case we have compatible partitioners, we can simply use the provided
  // vectors
  if(src.get_partitioner().get() == partitioner.get() &&
     dst.get_partitioner().get() == partitioner.get())
  {
    // This code is inspired to the code in TaskInfo::loop.
    if(overlap_communication_computation)
    {
      src.update_ghost_values_start(0);
      // In parallel, it's possible that some processors do not own any
      // cells.
      if(n_cells[0] > 0)
      {
        apply_kernel_shmem<dim, Number, Functor>
          <<<grid_dim[0], block_dim[0], Functor::shared_mem>>>(func,
                                                               get_cell_face_data(0),
                                                               src.get_values(),
                                                               dst.get_values());
        AssertCudaKernel();
      }
      src.update_ghost_values_finish();

      // In serial this color does not exist because there are no ghost
      // cells
      if(n_cells[1] > 0)
      {
        apply_kernel_shmem<dim, Number, Functor>
          <<<grid_dim[1], block_dim[1], Functor::shared_mem>>>(func,
                                                               get_cell_face_data(1),
                                                               src.get_values(),
                                                               dst.get_values());
        AssertCudaKernel();
        // We need a synchronization point because we don't want
        // CUDA-aware MPI to start the MPI communication until the
        // kernel is done.
        cudaDeviceSynchronize();
      }

      dst.compress_start(0, dealii::VectorOperation::add);
      // When the mesh is coarse it is possible that some processors do
      // not own any cells
      if(n_cells[2] > 0)
      {
        apply_kernel_shmem<dim, Number, Functor>
          <<<grid_dim[2], block_dim[2], Functor::shared_mem>>>(func,
                                                               get_cell_face_data(2),
                                                               src.get_values(),
                                                               dst.get_values());
        AssertCudaKernel();
      }
      dst.compress_finish(dealii::VectorOperation::add);
    }
    else
    {
      src.update_ghost_values();

      // Execute the loop on the cells
      for(unsigned int i = 0; i < n_colors; ++i)
        if(n_cells[i] > 0)
        {
          apply_kernel_shmem<dim, Number, Functor>
            <<<grid_dim[i], block_dim[i], Functor::shared_mem>>>(func,
                                                                 get_cell_face_data(i),
                                                                 src.get_values(),
                                                                 dst.get_values());
        }
      dst.compress(dealii::VectorOperation::add);
    }
    src.zero_out_ghost_values();
  }
  else
  {
    // Create the ghosted source and the ghosted destination
    VectorType ghosted_src(partitioner);
    VectorType ghosted_dst(ghosted_src);
    ghosted_src = src;
    ghosted_dst = dst;

    // Execute the loop on the cells
    for(unsigned int i = 0; i < n_colors; ++i)
      if(n_cells[i] > 0)
      {
        apply_kernel_shmem<dim, Number, Functor>
          <<<grid_dim[i], block_dim[i], Functor::shared_mem>>>(func,
                                                               get_cell_face_data(i),
                                                               ghosted_src.get_values(),
                                                               ghosted_dst.get_values());
        AssertCudaKernel();
      }

    // Add the ghosted values
    ghosted_dst.compress(dealii::VectorOperation::add);
    dst = ghosted_dst;
  }
}

} // namespace CUDAWrappers
} // namespace ExaDG
