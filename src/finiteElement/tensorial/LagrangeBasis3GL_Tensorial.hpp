#ifndef _LAGRANGEBASIS3GL_TENSORIAL_HPP_
#define _LAGRANGEBASIS3GL_TENSORIAL_HPP_

/**
 * @file LagrangeBasis3GL_Tensorial.hpp
 * @brief Tensorial implementation of Q3 Lagrange basis with Gauss-Lobatto points
 *
 * This implements an O(n^4) matrix-free stiffness operator using:
 * - Derivative matrices (D matrix)
 * - Kronecker product operations
 * - Precomputed geometric metrics
 */

#include <data_type.h>

/**
 * @class LagrangeBasis3GL_Tensorial
 * @brief Third-order Lagrange polynomial basis with Gauss-Lobatto quadrature points
 *
 * Parent space defined by:
 *                 o---------o--------o---------o  ---> xi
 *  Index:         0         1        2         3
 *  Coordinate:   -1    -1/sqrt(5) 1/sqrt(5)    1
 */
class LagrangeBasis3GL_Tensorial {
public:
  /// The number of support points for the basis
  constexpr static int numSupportPoints = 4;

  /// sqrt(5)
  constexpr static double sqrt5 = 2.2360679774997897;

private:
  // Static derivative matrix for O(n^4) algorithm
  // D[i][j] = d(phi_j)/d(xi) evaluated at xi_i
  constexpr static double D[4][4] = {
      {-3.0, 4.04508497187474, -1.54508497187474, 0.5},
      {-0.809016994374947, 0.0, 1.11803398874989, -0.309016994374947},
      {0.309016994374947, -1.11803398874989, 0.0, 0.809016994374947},
      {-0.5, 1.54508497187474, -4.04508497187474, 3.0}
  };

public:
  /**
   * @brief The value of the weight for the given support point
   * @param q The index of the support point
   * @return The value of the quadrature weight
   */
  PROXY_HOST_DEVICE
  constexpr static double weight(const int q) {
    switch (q) {
    case 1:
    case 2:
      return 5.0 / 6.0;
    default:
      return 1.0 / 6.0;
    }
  }

  /**
   * @brief Calculate the parent coordinates for the xi0 direction
   * @param supportPointIndex The linear index of support point
   * @return parent coordinate in the xi0 direction
   */
  PROXY_HOST_DEVICE
  constexpr static double parentSupportCoord(const int supportPointIndex) {
    double result = 0.0;

    switch (supportPointIndex) {
    case 0:
      result = -1.0;
      break;
    case 1:
      result = -1.0 / sqrt5;
      break;
    case 2:
      result = 1.0 / sqrt5;
      break;
    case 3:
      result = 1.0;
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @brief Access element (i,j) of the derivative matrix D
   * @param i Row index
   * @param j Column index
   * @return Value of D[i][j]
   *
   * The derivative matrix D is such that D[i][j] = d(phi_j)/d(xi) evaluated at xi_i
   * This is the key to the tensorial O(n^4) implementation.
   */
  PROXY_HOST_DEVICE PROXY_FORCE_INLINE
  constexpr static double derivativeMatrix(const int i, const int j) {
    return D[i][j];
  }

  /**
   * @brief Get diagonal entry of mass matrix (quadrature weight)
   * @param i Index of support point
   * @return Diagonal mass matrix entry
   */
  PROXY_HOST_DEVICE
  constexpr static double massDiagonal(const int i) {
    return weight(i);
  }

  /**
   * @brief The value of the basis function for a support point evaluated at a point
   * @param index The index of the support point
   * @param xi The coordinate at which to evaluate the basis
   * @return The value of basis function
   */
  PROXY_HOST_DEVICE
  constexpr static double value(const int index, const double xi) {
    double result = 0.0;

    switch (index) {
    case 0:
      result = -(5.0 / 8.0) * (xi * xi * xi - xi * xi - (1.0 / 5.0) * xi + 1.0 / 5.0);
      break;
    case 1:
      result = (5.0 * sqrt5 / 8.0) * (xi * xi * xi - (1.0 / sqrt5) * xi * xi - xi + 1.0 / sqrt5);
      break;
    case 2:
      result = -(5.0 * sqrt5 / 8.0) * (xi * xi * xi + (1.0 / sqrt5) * xi * xi - xi - 1.0 / sqrt5);
      break;
    case 3:
      result = (5.0 / 8.0) * (xi * xi * xi + xi * xi - (1.0 / 5.0) * xi - 1.0 / 5.0);
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @brief The gradient of the basis function for a support point
   * @param index The index of the support point
   * @param xi The coordinate at which to evaluate the gradient
   * @return The value of the gradient
   */
  PROXY_HOST_DEVICE
  constexpr static double gradient(const int index, const double xi) {
    double result = 0.0;

    switch (index) {
    case 0:
      result = -(5.0 / 8.0) * (3.0 * xi * xi - 2.0 * xi - (1.0 / 5.0));
      break;
    case 1:
      result = (5.0 * sqrt5 / 8.0) * (3.0 * xi * xi - (2.0 / sqrt5) * xi - 1.0);
      break;
    case 2:
      result = -(5.0 * sqrt5 / 8.0) * (3.0 * xi * xi + (2.0 / sqrt5) * xi - 1.0);
      break;
    case 3:
      result = (5.0 / 8.0) * (3.0 * xi * xi + 2.0 * xi - (1.0 / 5.0));
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @class TensorProduct3D
   * @brief 3D tensor product space for Q3 elements
   */
  struct TensorProduct3D {
    /// The number of support points in the 3D tensor product (4^3 = 64)
    constexpr static int numSupportPoints = 64;

    /**
     * @brief Calculates the linear index from ijk coordinates
     * @param i The index in the xi0 direction (0-3)
     * @param j The index in the xi1 direction (0-3)
     * @param k The index in the xi2 direction (0-3)
     * @return The linear index (0-63)
     */
    PROXY_HOST_DEVICE
    constexpr static int linearIndex(const int i, const int j, const int k) {
      return i + 4 * j + 16 * k;
    }

    /**
     * @brief Calculate the Cartesian/TensorProduct index from linear index
     * @param linearIndex The linear index of support point
     * @param i0 The Cartesian index in the xi0 direction
     * @param i1 The Cartesian index in the xi1 direction
     * @param i2 The Cartesian index in the xi2 direction
     */
    PROXY_HOST_DEVICE
    constexpr static void multiIndex(int const linearIndex, int &i0, int &i1, int &i2) {
      i2 = linearIndex / 16;
      i1 = (linearIndex % 16) / 4;
      i0 = (linearIndex % 16) % 4;
    }

    /**
     * @brief Evaluate all basis functions at a point
     * @param coords The coordinates (in parent frame) at which to evaluate
     * @param N Array to hold the value of the basis functions
     */
    PROXY_HOST_DEVICE
    static void value(const double (&coords)[3], double (&N)[numSupportPoints]) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          for (int c = 0; c < 4; ++c) {
            const int lindex = linearIndex(a, b, c);
            N[lindex] = LagrangeBasis3GL_Tensorial::value(a, coords[0]) *
                        LagrangeBasis3GL_Tensorial::value(b, coords[1]) *
                        LagrangeBasis3GL_Tensorial::value(c, coords[2]);
          }
        }
      }
    }
  };
};

#endif /* _LAGRANGEBASIS3GL_TENSORIAL_HPP_ */
