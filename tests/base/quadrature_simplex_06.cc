// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// integrates the function *f(x,y)/R, where f(x,y) is a power of x and
// y on the set [0,1]x[0,1]. dim = 2 only.

#include <deal.II/base/utilities.h>

#include "../tests.h"

// all include files needed for the program
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>

#include "simplex.h"


int
main()
{
  initlog();

  deallog << std::endl
          << "Calculation of the integral of f(x,y)*1/R on [0,1]x[0,1]"
          << std::endl
          << "for f(x,y) = x^i y^j, with i,j ranging from 0 to 5, and R being"
          << std::endl
          << "the distance from (x,y) to four vertices of the square."
          << std::endl
          << std::endl;

  double eps = 1e-10;

  //       index  m  i  j
  double error[4][6][6][6] = {{{{0}}}};

  for (unsigned int m = 0; m < 6; ++m)
    {
      for (unsigned int index = 0; index < 4; ++index)
        {
          auto split_point = GeometryInfo<2>::unit_cell_vertex(index);

          QSplit<2> quad(QTrianglePolar(m + 1), split_point);

          for (unsigned int i = 0; i < 6; ++i)
            for (unsigned int j = 0; j < 6; ++j)
              {
                double exact_integral  = exact_integral_one_over_r(index, i, j);
                double approx_integral = 0;

                for (unsigned int q = 0; q < quad.size(); ++q)
                  {
                    double x = quad.point(q)[0];
                    double y = quad.point(q)[1];
                    approx_integral +=
                      (pow(x, (double)i) * pow(y, (double)j) * quad.weight(q) /
                       (quad.point(q) - split_point).norm());
                  }
                error[index][m][i][j] = approx_integral - exact_integral;
              }
        }
    }


  // Now output the results.
  for (unsigned int index = 0; index < 4; ++index)
    {
      deallog << " ===============Vertex Index: " << index
              << " ==============================" << std::endl;
      for (unsigned int i = 0; i < 6; ++i)
        for (unsigned int j = 0; j < 6; ++j)
          {
            deallog << "======= f(x,y) = x^" << i << " y^" << j << std::endl;

            for (unsigned int m = 0; m < 6; ++m)
              deallog << "Order[" << m + 1
                      << "], error = " << error[index][m][i][j] << std::endl;
          }
    }
}