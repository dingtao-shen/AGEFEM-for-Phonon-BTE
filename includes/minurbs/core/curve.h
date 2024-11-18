/**
 * The Curve class represents a non-uniform polynomial B-spline curve, while the
 * RationalCurve class represents a non-uniform rational B-spline (NURBS) curve.
 *
 */

#ifndef MINURBS_CURVE
#define MINURBS_CURVE

#include "../../Eigen/Dense"
#include <assert.h>
#include <exception>
#include <stdexcept>
#include <cmath>

namespace minurbs{

    // Forward declaration
    template <typename T> struct RationalCurve;

    /**
    Struct for holding a polynomial B-spline curve
    @tparam T Data type of control points and knots (float or double)
    */
    template <typename T=double> struct Curve
    {
        unsigned int degree;
        Eigen::VectorXd knots;
        Eigen::MatrixXd control_points; // Np rows by 2 col

        Curve() = default;
        Curve(const RationalCurve<T> &crv) : Curve(crv.degree, crv.knots, crv.control_points) {}
        // Curve(RationalCurve crv) : Curve(crv.degree, crv.knots, crv.control_points) {}
        Curve(unsigned int degree, Eigen::VectorXd knots, Eigen::MatrixXd control_points)
            : degree(degree), knots(knots), control_points(control_points)
        {
        }
    };

    /**
    Struct for holding a rational B-spline curve
    @tparam T Data type of control points and knots (float or double)
    */
    template <typename T=double> struct RationalCurve
    {
        unsigned int degree;
        Eigen::VectorXd knots;
        Eigen::MatrixXd control_points;
        Eigen::VectorXd weights;

        RationalCurve() = default;
        RationalCurve(const Curve<T> &crv): RationalCurve(crv, Eigen::VectorXd::Ones(crv.control_points.size()))
        {
        }
        RationalCurve(const Curve<T> &crv, Eigen::VectorXd weights)
            : RationalCurve(crv.degree, crv.knots, crv.control_points, weights)
        {
        }
        RationalCurve(unsigned int degree, Eigen::VectorXd knots, Eigen::MatrixXd control_points, Eigen::VectorXd weights)
            : degree(degree), knots(knots), control_points(control_points), weights(weights)
        {
        }
    };


} // namespace minurbs


#endif