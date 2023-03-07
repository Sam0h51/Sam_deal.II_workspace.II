/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
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
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>


namespace Step26
{
  using namespace dealii;


  template <int dim, int spacedim>
  class HeatEquation
  {
  public:
    HeatEquation(const double ts = 1./500);
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

    void generate_mesh();
    void hyper_L();
    void sphere();
    void torus();

    Triangulation<dim, spacedim> triangulation;
    FE_Q<dim, spacedim>          fe;
    DoFHandler<dim, spacedim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    double       time;
    double       time_step;
    unsigned int timestep_number;

    const double theta;
  };




  template <int spacedim>
  class RightHandSide : public Function<spacedim>
  {
  public:
    RightHandSide()
      : Function<spacedim>()
      , period(0.1)
    {}

    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;

  private:
    const double period;
  };



  template <int spacedim>
  double RightHandSide<spacedim>::value(const Point<spacedim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    // Assert(dim == 2, ExcNotImplemented());

    const double time = this->get_time();
    const double point_within_period =
      ((time / period) - std::floor(time / period));

    /* std::cout << 2*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::cos(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::cos(2*3.14159265358*point_within_period)
              << std::endl
              << 4*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::cos(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::cos(2*3.14159265358*point_within_period)
              << std::endl
              << 2*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::cos(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::cos(2*3.14159265358*point_within_period)
              << std::endl
              << 4*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::sin(2*3.14159265358*point_within_period)
              << "\t"
              << 4*std::cos(2*3.14159265358*point_within_period)
              << "\t"
              << 2*std::cos(2*3.14159265358*point_within_period)
              << std::endl << std::endl; */


    double theta = 2*3.14159265358*point_within_period;

    if(point_within_period <= .125 || point_within_period >= .875){
      if(std::abs(p(2) - (std::tan(theta)*(p(0) - 2*std::cos(theta)) + 2*std::sin(theta))) <= 1e-1 &&
         p(0) >= 0){
        return 1;
      }
      else if(std::abs(p(2) - (std::tan(theta)*(p(0) - 2*std::cos(theta)) + 2*std::sin(theta))) <= 1e-1 &&
         p(0) <= 0){
        return -.1;
      }
      else{
        return -.1;
      }
    }
    else if(point_within_period >= .125 && point_within_period <= .375){
      if(std::abs(p(0) - ((std::cos(theta)/std::sin(theta))*(p(2) - 2*std::sin(theta)) + 2*std::cos(theta))) <= 1e-1
         && p(2) >= 0){
        return 1;
      }
      else if(std::abs(p(0) - ((std::cos(theta)/std::sin(theta))*(p(2) - 2*std::sin(theta)) + 2*std::cos(theta))) <= 1e-1
         && p(2) <= 0){
        return -.1;
      }
      else{
        return -.1;
      }
    }
    else if(point_within_period >= .375 && point_within_period <= .625){
      if(std::abs(p(2) - (std::tan(theta)*(p(0) - 2*std::cos(theta)) + 2*std::sin(theta))) <= 1e-1 &&
         p(0) <= 0){
        return 1;
      }
      else if(std::abs(p(2) - (std::tan(theta)*(p(0) - 2*std::cos(theta)) + 2*std::sin(theta))) <= 1e-1 &&
         p(0) >= 0){
        return -.1;
      }
      else{
        return -.1;
      }
    }
    else{
      if(std::abs(p(0) - ((std::cos(theta)/std::sin(theta))*(p(2) - 2*std::sin(theta)) + 2*std::cos(theta))) <= 1e-1
         && p(2) <= 0){
        return 1;
      }
      else if(std::abs(p(0) - ((std::cos(theta)/std::sin(theta))*(p(2) - 2*std::sin(theta)) + 2*std::cos(theta))) <= 1e-1
         && p(2) >= 0){
        return -.1;
      }
      else{
        return -.1;
      }
    }

    
    /* if(point_within_period <= .25)
    {
      if ((1.5*std::sin(2*3.14159265358*point_within_period) <= p(0)) &&
        (4.5*std::sin(2*3.14159265358*point_within_period) >= p(0)) &&
        (1.5*std::cos(2*3.14159265358*point_within_period) <= p(2)) &&
        (4.5*std::cos(2*3.14159265358*point_within_period) >= p(2))){
          return 1;
        }
      else
        return -.05;
    }
    else if(point_within_period >=.25 && point_within_period <=.5){
      if ((1.5*std::sin(2*3.14159265358*point_within_period) <= p(0)) &&
        (4.5*std::sin(2*3.14159265358*point_within_period) >= p(0)) &&
        (4.5*std::cos(2*3.14159265358*point_within_period) <= p(2)) &&
        (1.5*std::cos(2*3.14159265358*point_within_period) >= p(2))){
          return 1;
        }
      else
        return -.05;
    }
    else if (point_within_period >= .5 && point_within_period <= .75){
      if ((4.5*std::sin(2*3.14159265358*point_within_period) <= p(0)) &&
        (1.5*std::sin(2*3.14159265358*point_within_period) >= p(0)) &&
        (4.5*std::cos(2*3.14159265358*point_within_period) <= p(2)) &&
        (1.5*std::cos(2*3.14159265358*point_within_period) >= p(2))){
          return 1;
        }
      else
        return -.05;
    }
    else{
      if ((4.5*std::sin(2*3.14159265358*point_within_period) <= p(0)) &&
        (1.5*std::sin(2*3.14159265358*point_within_period) >= p(0)) &&
        (1.5*std::cos(2*3.14159265358*point_within_period) <= p(2)) &&
        (4.5*std::cos(2*3.14159265358*point_within_period) >= p(2))){
          return 1;
        }
      else
        return -.05;
    }; */


    /* if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 1) && (p[2] > 1))
          return 1;
        else
          return -1;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] < -1) && (p[2] < -1))
          return 1;
        else
          return -1;
      }
    else
      return 0; */
  }



  template <int spacedim>
  class BoundaryValues : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int spacedim>
  double BoundaryValues<spacedim>::value(const Point<spacedim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }



  template <int dim, int spacedim>
  HeatEquation<dim, spacedim>::HeatEquation(const double ts)
    : fe(1)
    , dof_handler(triangulation)
    , time_step(ts)
    , theta(0.5)
  {}




template<int dim, int spacedim>
void HeatEquation<dim, spacedim>::generate_mesh()
{
  torus();
}


template<int dim, int spacedim>
void HeatEquation<dim, spacedim>::hyper_L()
{
  GridGenerator::hyper_L(triangulation);
}

template<int dim, int spacedim>
void HeatEquation<dim, spacedim>::sphere()
{
  const Point<spacedim> center(0, 0, 0);
  const double radius = 1;

  GridGenerator::hyper_sphere(triangulation, center, radius);
}


template<int dim, int spacedim>
void HeatEquation<dim, spacedim>::torus()
{
  const double outer_radius = 3, inner_radius = 1;
  const unsigned int number_of_cells = 6;

  GridGenerator::torus(triangulation, outer_radius, inner_radius, number_of_cells);
}



























  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::solve_time_step()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }



  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::output_results() const
  {
    DataOut<dim, spacedim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U", DataOut<dim, spacedim>::type_dof_data);

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }


  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim, spacedim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<spacedim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.5,
                                                      0.5);

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<dim, Vector<double>, spacedim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement();
    setup_system();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);
  }



  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::run()
  {
    const unsigned int initial_global_refinement       = 6;
    const unsigned int n_adaptive_pre_refinement_steps = 6;

    generate_mesh();
    triangulation.refine_global(initial_global_refinement);

    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    Vector<double> forcing_terms;

  start_time_iteration:

    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());


    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<spacedim>(),
                             old_solution);
    solution = old_solution;

    output_results();

    while (time <= 0.5)
      {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step, tmp);

        RightHandSide<spacedim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= time_step * theta;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree + 1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add(time_step * (1 - theta), tmp);

        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);

        constraints.condense(system_matrix, system_rhs);

        {
          BoundaryValues<spacedim> boundary_values_function;
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }

        solve_time_step();

        output_results();

        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;

            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());

            std::cout << std::endl;

            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
          }

        old_solution = solution;
      }
  }
} // namespace Step26


int main()
{
  try
    {
      using namespace Step26;

      HeatEquation<2, 3> heat_equation_solver;
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
