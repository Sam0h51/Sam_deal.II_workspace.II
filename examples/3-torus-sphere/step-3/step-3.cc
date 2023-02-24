/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace dealii;



class Step3
{
public:
  Step3();

  void run_torus();
  void run_sphere();


private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(std::string name) const;

  double rhs_function(const Point<3> &p);

  void make_torus();
  void make_sphere();

  Triangulation<2, 3> triangulation;
  FE_Q<2, 3>          fe;
  DoFHandler<2, 3>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}


void Step3::make_torus()
{
  const double inner_rad = 2, outer_rad = 3;
  const unsigned int n_cells = 6;

  GridGenerator::torus(triangulation, outer_rad, inner_rad, n_cells);
  triangulation.refine_global(6);
}

void Step3::make_sphere()
{
  const Point<3> center(0, 0, 0);
  const double radius = 1;          //Note: Radius defaults to 1 for sphere, so this is unnecessary

  GridGenerator::hyper_sphere(triangulation, center, radius);
  triangulation.refine_global(6);
}


double Step3::rhs_function(const Point<3> &p)
{
  if(p(1) >= 1.6){
    return 20 + 80*std::pow(std::sin(2*p[0]),4)*std::pow(std::sin(2*p[2]),4);
  }
  else{
    return 2;
  }
}






void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2, 3> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices()){
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

              cell_matrix(i, j) +=
                (fe_values.shape_value(i, q_index) *
                 fe_values.shape_value(j, q_index) *
                 fe_values.JxW(q_index)
                );
            };

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            rhs_function(fe_values.quadrature_point(q_index)) *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}



void Step3::solve()
{
  SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}



void Step3::output_results(std::string name) const
{
  DataOut<2, 3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, name, DataOut<2, 3>::type_dof_data);
  data_out.build_patches();

  std::string out_name = name + ".vtk";

  std::ofstream output(out_name);
  data_out.write_vtk(output);
}



void Step3::run_torus()
{
  make_torus();
  setup_system();
  assemble_system();
  solve();
  output_results("torus");
}

void Step3::run_sphere()
{
  make_sphere();
  setup_system();
  assemble_system();
  solve();
  output_results("sphere");
}



int main()
{
  deallog.depth_console(2);

  {
    Step3 torus;
    torus.run_torus();
  }

  {
    Step3 sphere;
    sphere.run_sphere();
  }

  return 0;
}
