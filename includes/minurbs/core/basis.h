/* 
* Low-level functions for evaluating B-spline basis functions and their derivatives
* Double precision are used throughout all codes
*/

#ifndef MINURBS_BASIS
#define MINURBS_BASIS

#include "../../Eigen/Dense"
#include <assert.h>
#include <cmath>

namespace minurbs
{
    
/**
 * Find the span of the given parameter in the knot vector.
 * @param[in] deg Degree of the curve.
 * @param[in] knots Knot vector of the curve.
 * @param[in] u Parameter value.
 * @return Span index into the knot vector such that (span - 1) < u <= span
*/
template <typename T=double> int findSpan(unsigned int deg, Eigen::VectorXd knots, double u)
{
    // index of last control point
    int n = static_cast<int>(knots.size()) - deg - 2;
    assert(n >= 0);

    // For values of u that lies outside the domain
    if (u >= knots(n + 1))
    {
        return n;
    }
    if (u <= knots(deg))
    {
        return deg;
    }

    // Binary search
    int low = deg;
    int high = n + 1;
    int mid = (int)std::floor((low + high) / 2.0);
    while (u < knots(mid) || u >= knots(mid + 1))
    {
        if (u < knots(mid))
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
        mid = (int)std::floor((low + high) / 2.0);
    }
    return mid;
}

/**
 * Compute all nonvanishing B-spline basis functions
 * @param[in] deg Degree of the basis function.
 * @param[in] span Index obtained from findSpan() corresponding the u and knots.
 * @param[in] knots Knot vector corresponding to the basis functions.
 * @param[in] u Parameter to evaluate the basis functions at.
 * @return N Values of (deg+1) non-zero basis functions.
 */
template <typename T=double> Eigen::VectorXd bsplineBasis(unsigned int deg, int span, Eigen::VectorXd knots, double u)
{
    Eigen::VectorXd N = Eigen::VectorXd::Zero(deg + 1);
    Eigen::VectorXd left = Eigen::VectorXd::Zero(deg + 1);
    Eigen::VectorXd right = Eigen::VectorXd::Zero(deg + 1);
    double saved = 0.0, temp = 0.0;

    N(0) = 1.0;

    for (int j = 1; j <= deg; j++)
    {
        left(j) = (u - knots(span + 1 - j));
        right(j) = knots(span + j) - u;
        saved = 0.0;
        for (int r = 0; r < j; r++)
        {
            temp = N(r) / (right(r + 1) + left(j - r));
            N(r) = saved + right(r + 1) * temp;
            saved = left(j - r) * temp;
        }
        N(j) = saved;
    }
    return N;
}

/**
 * Compute all nonvanishing derivatives of B-spline basis functions
 * @param[in] deg Degree of the basis function.
 * @param[in] span Index obtained from findSpan() corresponding the u and knots.
 * @param[in] knots Knot vector corresponding to the basis functions.
 * @param[in] u Parameter to evaluate the basis functions at.
 * @param[in] num_ders Number of derivatives to compute (num_ders <= deg)
 * @return ders Values of non-zero derivatives of basis functions.
 */
template <typename T=double> Eigen::MatrixXd bsplineDerBasis(unsigned int deg, int span, Eigen::VectorXd knots, double u, int num_ders)
{
    Eigen::VectorXd left = Eigen::VectorXd::Zero(deg + 1);
    Eigen::VectorXd right = Eigen::VectorXd::Zero(deg + 1);
    double saved = 0.0, temp = 0.0;

    Eigen::MatrixXd ndu = Eigen::MatrixXd::Zero(deg + 1, deg + 1);
    ndu(0, 0) = 1.0;

    for (int j = 1; j <= deg; j++)
    {
        left(j) = u - knots(span + 1 - j);
        right(j) = knots(span + j) - u;
        saved = 0.0;

        for (int r = 0; r < j; r++)
        {
            // Lower triangle
            ndu(j, r) = right(r + 1) + left(j - r);
            temp = ndu(r, j - 1) / ndu(j, r);
            // Upper triangle
            ndu(r, j) = saved + right(r + 1) * temp;
            saved = left(j - r) * temp;
        }

        ndu(j, j) = saved;
    }

    Eigen::MatrixXd ders = Eigen::MatrixXd::Zero(num_ders + 1, deg + 1);

    for (int j = 0; j <= deg; j++)
    {
        ders(0, j) = ndu(j, deg);
    }

    Eigen::MatrixXd a(2, deg + 1);

    for (int r = 0; r <= deg; r++)
    {
        int s1 = 0;
        int s2 = 1;
        a(0, 0) = 1.0;

        for (int k = 1; k <= num_ders; k++)
        {
            double d = 0.0;
            int rk = r - k;
            int pk = deg - k;
            int j1 = 0;
            int j2 = 0;

            if (r >= k)
            {
                a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                d = a(s2, 0) * ndu(rk, pk);
            }

            if (rk >= -1)
            {
                j1 = 1;
            }
            else
            {
                j1 = -rk;
            }

            if (r - 1 <= pk)
            {
                j2 = k - 1;
            }
            else
            {
                j2 = deg - r;
            }

            for (int j = j1; j <= j2; j++)
            {
                a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                d += a(s2, j) * ndu(rk + j, pk);
            }

            if (r <= pk)
            {
                a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                d += a(s2, k) * ndu(r, pk);
            }

            ders(k, r) = d;

            int temp = s1;
            s1 = s2;
            s2 = temp;
        }
    }

    double fac = deg;
    for (int k = 1; k <= num_ders; k++)
    {
        for (int j = 0; j <= deg; j++)
        {
            ders(k, j) *= fac;
        }
        fac *= double(deg - k);
    }

    return ders;
}


} // namespace minurbs



#endif