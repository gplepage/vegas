c Sample vegas integrand written in Fortran

      function fcn(x, dim)
      integer i, dim
      real*8 x(dim), xsq, fcn
      xsq = 0.0
      do i=1,dim
        xsq = xsq + x(i) ** 2
      end do
      fcn = exp(-100. * sqrt(xsq)) * 100. ** dim
      return
      end

c Batch form of integrand.

      subroutine batch_fcn(ans, x, dim, nbatch)
      integer dim, nbatch, i, j
      real*8 x(nbatch, dim), xi(dim), ans(nbatch), fcn
cf2py intent(out) ans
      do i=1,nbatch
            do j=1,dim
                  xi(j) = x(i, j)
            end do
            ans(i) = fcn(xi, dim)
      end do
      end

c Copyright (c) 2016-18 G. Peter Lepage.
c
c This program is free software: you can redistribute it and/or modify
c it under the terms of the GNU General Public License as published by
c the Free Software Foundation, either version 3 of the License, or
c any later version (see <http://www.gnu.org/licenses/>).
c
c This program is distributed in the hope that it will be useful,
c but WITHOUT ANY WARRANTY; without even the implied warranty of
c MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c GNU General Public License for more details.
