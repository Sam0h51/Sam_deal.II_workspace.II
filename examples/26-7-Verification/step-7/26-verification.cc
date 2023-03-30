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
    HeatEquation();
    HeatEquation(double time_step_denominator
                , unsigned int ref_num
                , double r_constant = 0.5
                , double g1_constant = 0.5
                , double k_constant = 1.);
    void run();

    void test_matrices();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);

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
    unsigned int timestep_denominator;
    unsigned int refinement_number;

    const double theta;
    const double r;
    const double g1;
    const double k;
  };






  template <int spacedim>
  class SolutionBase
  {
  public:
    SolutionBase(double r_constant, double g_1_constant, double tau_constant, double width_constant = 1./8.);
  protected:
    static const std::array<Point<spacedim>, 1> source_centers;
    const double                    width;
    const double                    r;
    const double                    g_1;
    const double                    tau;
  };

  template<int spacedim>
  SolutionBase<spacedim>::SolutionBase(double r_constant, double g_1_constant, double tau_constant, double width_constant)
    : r(r_constant), g_1(g_1_constant), tau(tau_constant), width(width_constant)
  {}

  template <>
  const std::array<Point<1>, 1> SolutionBase<1>::source_centers = {
    {Point<1>(-1.0 / 3.0)}};

  template <>
  const std::array<Point<2>, 1> SolutionBase<2>::source_centers = {
    {Point<2>(-0.5, +0.5)}};



  template <int dim>
  class Solution : public Function<dim>, public SolutionBase<dim>
  {
  public:
    Solution(double r_constant, double g_1_constant, double tau_constant);

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;
  };

  template<int dim>
  Solution<dim>::Solution(double r_constant, double g_1_constant, double tau_constant)
    : SolutionBase<dim>(r_constant, g_1_constant, tau_constant)
  {}

  template <int dim>
  double Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    double return_value = 0;

    double t = this->get_time();

    for (const auto &center : this->source_centers)
      {
        const Tensor<1, dim> x_minus_xi = p - center;
        return_value +=
          std::exp(this->tau*t) *                                               //time component of exact solution
          std::exp(-x_minus_xi.norm_square() / (this->width * this->width));    //shape component of exact solution
      }

    return return_value;
  }


  //Depricated: DO NOT USE!!
  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                         const unsigned int) const
  {
    Tensor<1, dim> return_value;

    for (const auto &center : this->source_centers)
      {
        const Tensor<1, dim> x_minus_xi = p - center;

        return_value +=
          (-2. / (this->width * this->width) *
           std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) *
           x_minus_xi);
      }

    return return_value;
  }

  template <int dim>
  class RightHandSide : public Function<dim>, public SolutionBase<dim>
  {
  public:
    RightHandSide(double r_constant, double g_1_constant, double tau_constant);

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template<int dim>
  RightHandSide<dim>::RightHandSide(double r_constant, double g_1_constant, double tau_constant)
    : SolutionBase<dim>(r_constant, g_1_constant, tau_constant)
  {}


  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p,
                                   const unsigned int) const
  {
    double return_value = 0;

    double tau = this->tau;
    double r   = this->r;
    double g_1 = this->g_1;


    for (const auto &center : this->source_centers)
      {
        const Tensor<1, dim> x_minus_xi = p - center;

        const double t = this->get_time();

        //Laplace u
        return_value +=
          ((2. * dim -
            4. * x_minus_xi.norm_square() / (this->width * this->width)) /
           (this->width * this->width) *
           std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * 
           std::exp(tau*t));

        //(1-r)u
        return_value +=
          (1-r)*std::exp(tau*t)*std::exp(-x_minus_xi.norm_square() / (this->width * this->width));

        //Time derivative of u
        return_value +=
          tau*std::exp(tau*t)*std::exp(-x_minus_xi.norm_square()/(this->width * this->width));

        //Square term
        return_value +=
          -g_1*std::exp(2*tau*t)*std::exp(-2*x_minus_xi.norm_square()/(this->width * this->width));

        //Cubic term
        return_value +=
          std::exp(3*tau*t)*std::exp(-3*x_minus_xi.norm_square()/(this->width * this->width));
      }

    return return_value;
  }




















/*   template <int spacedim>
  class RightHandSide : public Function<spacedim>
  {
  public:
    RightHandSide()
      : Function<spacedim>()
      , period(0.2)
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

    const double time = this->get_time();
    const double point_within_period =
      (time / period - std::floor(time / period));

    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  } */



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

  template<int spacedim>
  class InitialConditions : public Function<spacedim>
  {
  private:
    /* data */
  public:

    virtual double value(const Point<spacedim> &p,
                         const unsigned int component = 0) const override;
  };

  template<int spacedim>
  double InitialConditions<spacedim>::value(const Point<spacedim> &p,
                                     const unsigned int component) const
  {
    /*NOTE: This code is here just so I don't get warnings
            about an unused componenet*/
    (void)component;
    AssertIndexRange(component, 1);

    // return 2.;


    double rad_1 = std::pow((p(0) - 4), 2) + std::pow(p(1), 2) + std::pow(p(2), 2);
    double rad_2 = std::pow((p(0) + 4), 2) + std::pow(p(1), 2) + std::pow(p(2), 2);
    if(rad_1 < 1e-1){
      return 1.;
    }

    if(rad_2 < 1e-1){
      return -1.;
    }

    return 0;
  }
  
















































  template <int dim, int spacedim>
  HeatEquation<dim, spacedim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / 1500)
    , timestep_denominator(1500)
    , refinement_number(4)
    , theta(0.5)
    , r(0.5)
    , g1(0.5)
    , k(1.)
  {}

  template <int dim, int spacedim>
  HeatEquation<dim, spacedim>::HeatEquation(double       time_step_denominator,
                                            unsigned int ref_num,
                                            double       r_constant,
                                            double       g1_constant,
                                            double       k_constant)
    : fe(1)
    , dof_handler(triangulation)
    , time_step(1. / time_step_denominator)
    , timestep_denominator(time_step_denominator)
    , refinement_number(ref_num)
    , theta(0.5)
    , r(r_constant)
    , g1(g1_constant)
    , k(k_constant)
  {}















































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

    std::string fraction_part = "-TimeStep-1-over-";

    if(timestep_denominator - 1000 < 0){
      fraction_part += "0";
    }

    const std::string filename =
      /* "Refinement-" + Utilities::int_to_string(refinement_number) 
                    + "-TimeStep-1-over-" + Utilities::int_to_string((int)timestep_denominator) 
                    +  */"solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
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
                                                      0.6,
                                                      0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    SolutionTransfer<dim, dealii::Vector<double>, spacedim> solution_trans(dof_handler);

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
    // Vector<double> convergence_vector;

    // GridGenerator::torus(triangulation, 3., 1.);
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refinement_number);

    setup_system();

    Vector<double> tmp;
    Vector<double> forcing_terms;

    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());

    // InitialConditions<spacedim> initial_conditions;
    Solution<spacedim> initial_conditions(this->r, this->g1, -0.1);

    initial_conditions.set_time(0.);

    std::cout << "Before interpolate" << std::endl;

    VectorTools::interpolate(dof_handler,
                             initial_conditions,
                             old_solution);

    std::cout << "After interpolate" << std::endl;

    solution = old_solution;

    output_results();

    while (time <= 0.5)
      {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        std::cout << "Vector is";
        std::cout << solution[3] << std::endl;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(-1*time_step, laplace_matrix);
        system_matrix.add((time_step - r*time_step), mass_matrix);

        const QGauss<dim> quadrature_formula(fe.degree + 1);

        //Copied from step-7
        const unsigned int n_q_points      = quadrature_formula.size();

        FEValues<dim, spacedim> fe_values(fe, quadrature_formula, 
                                          update_values | update_gradients |
                                          update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        //Copied from step-7
        RightHandSide<spacedim> right_hand_side(this->r, this->g1, -0.1);
        std::vector<double>      rhs_values(n_q_points);
        Solution<spacedim> exact_solution(this->r, this->g1, -0.1);

        right_hand_side.set_time(time);
        exact_solution.set_time(time);

        double Un1 = 0;

        // std::cout << std::endl << std::endl << "System RHS: " << system_rhs << std::endl << std::endl;

        system_rhs = 0;

        for(const auto &cell : dof_handler.active_cell_iterators()){
          cell_matrix = 0.;
          cell_rhs = 0.;

          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          //Copied from step-7
          right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);

          for(const unsigned int q_index : fe_values.quadrature_point_indices()){
            Un1 = 0;
            for(const unsigned int i : fe_values.dof_indices()){
              Un1 += old_solution(local_dof_indices[i])*fe_values.shape_value(i, q_index);
            }

            // std::cout << std::endl << std::endl << "Un1: " << Un1 << std::endl << std::endl;

            for(const unsigned int i : fe_values.dof_indices()){
              cell_rhs(i) += (Un1 + time_step*g1*std::pow(Un1, 2) - time_step*k*std::pow(Un1, 3))
                              *fe_values.shape_value(i, q_index)*fe_values.JxW(q_index);

              //Copied from step-7
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              rhs_values[q_index] *               // f(x_q)
                              fe_values.JxW(q_index));            // dx
            }

            // std::cout << std::endl << cell_rhs << std::endl;
          }

          for(unsigned int i : fe_values.dof_indices()){
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }


        }

        // std::cout << system_rhs << std::endl << std::endl;

        {
          Solution<spacedim> boundary_values_function(this->r, this->g1, -0.1);
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;

          //NOTE: The Solution object here is a new object, make sure the constructor variables match the
          //      variables given to exact_solution
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,   //Changed according to step-7
                                                   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }

        // std::cout << system_rhs << std::endl << std::endl << std::endl << std::endl;

        solve_time_step();

        output_results();

        /* convergence_vector.reinit(solution.size());

        convergence_vector+= solution;
        convergence_vector-= old_solution;

        std::cout << "Solution difference has magnitude " << convergence_vector.l2_norm() << std::endl;

        if(convergence_vector.l2_norm() < 1e-6){
          std::cout << std::endl << std::endl;
          std::cout << "Solution converged at step " << timestep_number;
          std::cout << std::endl << std::endl;
          break;
        } */

        /* if(timestep_number >= 300){
          std::cout << std::endl << std::endl;
          std::cout << "Breaking loop, expected convergence" << std::endl;
          std::cout << std::endl << std::endl;
          break;
        } */

        old_solution = solution;
      }
  }
  
  
  
  
  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::test_matrices()
  {
    Vector<double> tmp;

    GridGenerator::torus(triangulation, 3., 1.);
    // GridGenerator::hyper_cube(triangulation, -5, 5);
    triangulation.refine_global(refinement_number);

    InitialConditions<spacedim> initial_conditions;

    setup_system();

    tmp.reinit(solution.size());

    VectorTools::interpolate(dof_handler, initial_conditions, old_solution);

    laplace_matrix.vmult(tmp, old_solution);

    std::cout << tmp << std::endl << std::endl;

    tmp.reinit(solution.size());

    mass_matrix.vmult(tmp, old_solution);

    std::cout << tmp << std::endl << std::endl;

  }
} // namespace Step26


int main()
{
  try
    {
      using namespace Step26;

      HeatEquation<2, 2> heat_equation_solver(500, 3);
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
  /* for(unsigned int i = 0; i < 8; ++i){
    for(unsigned int j = 1; j<21; ++j){
      std::cout << std::endl;
      std::cout << "=============================================================" << std::endl;
      std::cout << "Running step-26 with refinement " << i << " and timestep 1 over " << j*200 << std::endl << std::endl;
      {
        try
          {
            using namespace Step26;

            HeatEquation<2, 2> heat_equation_solver(j*200, i);
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
      }
    }
  } */

  return 0;
}
