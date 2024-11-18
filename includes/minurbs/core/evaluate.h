/**
 * Core functionality for evaluating points, derivatives and related
 * quantities on NURBS curves and surfaces.
 *
 */

#ifndef TINYNURBS_EVALUATE
#define TINYNURBS_EVALUATE

#include <iostream>
using namespace std;
#include "../../Eigen/Dense"
#include "basis.h"
#include "curve.h"

/**
 * Compute the binomial coefficient (nCk) using the formula
 * \product_{i=0}^k (n + 1 - i) / i
 */
inline int binomial(unsigned int n, unsigned int k)
{
    unsigned int result = 1;
    if (k > n)
    {
        return 0;
    }
    for (unsigned int i = 1; i <= k; ++i)
    {
        result *= (n + 1 - i);
        result /= i;
    }
    return result;
}

namespace minurbs{

    namespace internal{

        /**
        * Evaluate point on a nonrational NURBS curve
        * @param[in] deg Degree of the given curve.
        * @param[in] knots Knot vector of the curve.
        * @param[in] control_points Control points of the curve.
        * @param[in] u Parameter to evaluate the curve at.
        * @return point Resulting point on the curve at parameter u.
        */
        template <typename T=double> Eigen::VectorXd curvePoint(unsigned int deg, Eigen::VectorXd knots, Eigen::MatrixXd control_points, double u)
        {
            // Initialize result to 0s
            Eigen::VectorXd point = Eigen::VectorXd::Zero(control_points.cols());

            // Find span and corresponding non-zero basis functions
            int span = findSpan(deg, knots, u);
            Eigen::VectorXd N = bsplineBasis(deg, span, knots, u);

            // Compute point
            for (int j = 0; j <= deg; j++)
            {
                for(int k = 0; k < point.size(); k++){
                    point(k) += double(N[j]) * control_points(span - deg + j, k);
                }
            }
            return point;
        }

        /**
         * Evaluate derivatives of a non-rational NURBS curve
         * @param[in] degree Degree of the curve
         * @param[in] knots Knot vector of the curve.
         * @param[in] control_points Control points of the curve.
         * @param[in] num_ders Number of times to derivate.
         * @param[in] u Parameter to evaluate the derivatives at.
         * @return curve_ders Derivatives of the curve at u.
         * E.g. curve_ders[n] is the nth derivative at u, where 0 <= n <= num_ders.
         */
        template <typename T=double> Eigen::MatrixXd curveDerivatives(unsigned int degree, Eigen::VectorXd knots, Eigen::MatrixXd control_points, double u, int num_ders)
        {
            Eigen::MatrixXd curve_ders = Eigen::MatrixXd::Zero(num_ders + 1, control_points.cols()); // Higher order derivatives are assigned to zero

            // Find the span and corresponding non-zero basis functions & derivatives
            int span = findSpan(degree, knots, u);
            Eigen::MatrixXd ders = bsplineDerBasis(degree, span, knots, u, num_ders);

            // Compute first num_ders derivatives
            int du = num_ders < degree ? num_ders : degree;
            for (int k = 0; k <= du; k++)
            {
                for (int j = 0; j <= degree; j++)
                {
                    for(int l = 0; l < curve_ders.cols(); l++){
                        curve_ders(k, l) += ders(k, j) * control_points(span - degree + j, l);
                    }
                }
            }
            return curve_ders;
        }

    } //namespace internal

    /**
    Evaluate point on a nonrational NURBS curve
    @param[in] crv Curve object
    @param[in] u Parameter to evaluate the curve at.
    @return point Resulting point on the curve at parameter u.
    */
    template <typename T=double> Eigen::VectorXd curvePoint(const Curve<T> &crv, double u)
    {
        return internal::curvePoint(crv.degree, crv.knots, crv.control_points, u);
    }

    /**
     * Evaluate point on a rational NURBS curve
     * @param[in] crv RationalCurve object
     * @param[in] u Parameter to evaluate the curve at.
     * @return point Resulting point on the curve.
     */
    template <typename T=double> Eigen::VectorXd curvePoint(const RationalCurve<T> &crv, double u)
    {
        // Convert cartesian coordinates to homogenous coordinates
        Eigen::MatrixXd Cw = Eigen::MatrixXd::Zero(crv.control_points.rows(), 3);
        for(int i = 0; i < crv.control_points.rows(); i++){
            Cw(i,0) = crv.control_points(i,0) * crv.weights(i);
            Cw(i,1) = crv.control_points(i,1) * crv.weights(i);
            Cw(i,2) = crv.weights(i);
        }
        // Compute point using homogenous coordinates
        Eigen::VectorXd pointw = internal::curvePoint(crv.degree, crv.knots, Cw, u);
        // Convert back to cartesian coordinates
        Eigen::VectorXd point(pointw.size()-1);
        double w = pointw(pointw.size()-1);
        for(int i = 0; i < point.size(); i++){
            point(i) = pointw(i) / w;
        }

        return point;
    }

    /**
     * Evaluate derivatives of a non-rational NURBS curve
     * @param[in] crv Curve object
     * @param[in] num_ders Number of times to derivate.
     * @param[in] u Parameter to evaluate the derivatives at.
     * @return curve_ders Derivatives of the curve at u.
     * E.g. curve_ders[n] is the nth derivative at u, where 0 <= n <= num_ders.
     */
    template <typename T=double> Eigen::MatrixXd curveDerivatives(const Curve<T> &crv, int num_ders, double u)
    {
        return internal::curveDerivatives(crv.degree, crv.knots, crv.control_points, u, num_ders);
    }

    /**
     * Evaluate derivatives of a rational NURBS curve
     * @param[in] u Parameter to evaluate the derivatives at.
     * @param[in] knots Knot vector of the curve.
     * @param[in] control_points Control points of the curve.
     * @param[in] weights Weights corresponding to each control point.
     * @param[in] num_ders Number of times to differentiate.
     * @param[inout] curve_ders Derivatives of the curve at u.
     * E.g. curve_ders[n] is the nth derivative at u, where n is between 0 and
     * num_ders-1.
     */
    template <typename T=double> Eigen::MatrixXd curveDerivatives(const RationalCurve<T> &crv, int num_ders, double u)
    {
        // Convert cartesian coordinates to homogenous coordinates
        Eigen::MatrixXd Cw = Eigen::MatrixXd::Zero(crv.control_points.rows(), 3);
        for(int i = 0; i < crv.control_points.rows(); i++){
            Cw(i,0) = crv.control_points(i,0) * crv.weights(i);
            Cw(i,1) = crv.control_points(i,1) * crv.weights(i);
            Cw(i,2) = crv.weights(i);
        }
        Eigen::MatrixXd curve_ders = Eigen::MatrixXd::Zero(num_ders + 1, Cw.cols()-1);
        // Derivatives of Cw
        Eigen::MatrixXd Cwders = internal::curveDerivatives(crv.degree, crv.knots, Cw, u, num_ders);

        // Split Cwders into coordinates and weights
        Eigen::MatrixXd Aders = Cwders.block(0, 0, Cwders.rows(), Cwders.cols()-1);
        Eigen::VectorXd wders = Cwders.block(0, Cwders.cols()-1, Cwders.rows(), 1);

        // Compute rational derivatives
        for (int k = 0; k <= num_ders; k++)
        {
            Eigen::VectorXd v = Aders.row(k);
            for (int i = 1; i <= k; i++)
            {
                v -= binomial(k, i) * wders(i) * curve_ders.row(k - i);
            }
            curve_ders.row(k) = v.array() /= wders(0);
        }
        return curve_ders;
    }

    /**
     * Evaluate the tangent of a B-spline curve
     * @param[in] crv Curve object
     * @return Unit tangent of the curve at u.
     */
    template <typename T=double> Eigen::VectorXd curveTangent(const Curve<T> &crv, double u)
    {
        Eigen::MatrixXd ders = curveDerivatives(crv, 1, u);
        Eigen::VectorXd du = ders.row(1);
        double du_len = sqrt(pow(du(0), 2) + pow(du(1), 2));
        if(du_len != 0){
            du.array() /= du_len;        
        }
        return du;
    }
    /**
     * Evaluate the tangent of a rational B-spline curve
     * @param[in] crv RationalCurve object
     * @return Unit tangent of the curve at u.
     */
    template <typename T=double> Eigen::VectorXd curveTangent(const RationalCurve<T> &crv, double u)
    {
        Eigen::MatrixXd ders = curveDerivatives(crv, 1, u);
        Eigen::VectorXd du = ders.row(1);
        double du_len = sqrt(pow(du(0), 2) + pow(du(1), 2));
        if(du_len != 0){
            du.array() /= du_len;        
        }
        return du;
    }


} // namespace minurbs

#endif