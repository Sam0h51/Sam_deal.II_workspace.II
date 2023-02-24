// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2020 by the deal.II authors
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


#include "../tests.h"

#include "fe_prolongation_common.h"



int
main()
{
  initlog();
  deallog << std::setprecision(9);

  CHECK_ALL(DGQ, 0, 2);
  CHECK_ALL(DGQ, 1, 2);
  CHECK_ALL(DGQ, 2, 2);
  CHECK_ALL(DGQ, 3, 2);
  CHECK_ALL(DGQ, 4, 2);

  CHECK_ALL(DGQ, 0, 3);
  CHECK_ALL(DGQ, 1, 3);
  CHECK_ALL(DGQ, 2, 3);
}