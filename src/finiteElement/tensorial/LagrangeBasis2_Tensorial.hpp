#ifndef _LAGRANGEBASIS2_TENSORIAL_HPP_
#define _LAGRANGEBASIS2_TENSORIAL_HPP_

/**
 * @file LagrangeBasis2_Tensorial.hpp
 * @brief Tensorial implementation of Q2 quadratic Lagrange basis
 *
 * This implements an O(n^4) matrix-free stiffness operator using:
 * - Derivative matrices (D matrix)
 * - Kronecker product operations
 * - Precomputed geometric metrics
 */

#include <data_type.h>

/**
 * @class LagrangeBasis2_Tensorial
 * @brief Second-order (quadratic) Lagrange polynomial basis
 *
 * Parent space defined by:
 *                 o-------------o-------------o  ---> xi
 *  Index:         0             1             2
 *  Coordinate:   -1             0             1
 */
class LagrangeBasis2_Tensorial {
public:
  /// The number of support points for the basis
  constexpr static int numSupportPoints = 3;

private:
  // Static derivative matrix for O(n^4) algorithm
  // D[i][j] = d(phi_j)/d(xi) evaluated at xi_i
  constexpr static double D[3][3] = {
      {-1.5, 2.0, -0.5},
      {-0.5, 0.0,  0.5},
      { 0.5, -2.0, 1.5}
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
    case 0:
    case 2:
      return 1.0 / 3.0;
    default:
      return 4.0 / 3.0;
    }
  }

  /**
   * @brief Calculate the parent coordinates for the xi0 direction
   * @param supportPointIndex The linear index of support point
   * @return parent coordinate in the xi0 direction
   */
  PROXY_HOST_DEVICE
  constexpr static double parentSupportCoord(const int supportPointIndex) {
    switch (supportPointIndex) {
    case 0:
      return -1.0;
    case 2:
      return 1.0;
    case 1:
    default:
      return 0.0;
    }
  }

  /**
   * @brief Access element (i,j) of the derivative matrix D
   * @param i Row index
   * @param j Column index
   * @return Value of D[i][j]
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
   * @brief The value of the basis function for a support point
   * @param index The index of the support point
   * @param xi The coordinate at which to evaluate the basis
   * @return The value of basis function
   */
  PROXY_HOST_DEVICE
  constexpr static double value(const int index, const double xi) {
    switch (index) {
    case 0:
      return -0.5 * xi + 0.5 * xi * xi;
    case 2:
      return 0.5 * xi + 0.5 * xi * xi;
    case 1:
    default:
      return 1.0 - xi * xi;
    }
  }

  /**
   * @brief The gradient of the basis function for a support point
   * @param index The index of the support point
   * @param xi The coordinate at which to evaluate the gradient
   * @return The value of the gradient
   */
  PROXY_HOST_DEVICE
  constexpr static double gradient(const int index, const double xi) {
    switch (index) {
    case 0:
      return -0.5 + xi;
    case 2:
      return 0.5 + xi;
    case 1:
    default:
      return -2.0 * xi;
    }
  }

  /**
   * @class TensorProduct3D
   * @brief 3D tensor product space for Q2 elements
   */
  struct TensorProduct3D {
    /// The number of support points in the 3D tensor product (3^3 = 27)
    constexpr static int numSupportPoints = 27;

    /**
     * @brief Calculates the linear index from ijk coordinates
     * @param i The index in the xi0 direction (0-2)
     * @param j The index in the xi1 direction (0-2)
     * @param k The index in the xi2 direction (0-2)
     * @return The linear index (0-26)
     */
    PROXY_HOST_DEVICE
    constexpr static int linearIndex(const int i, const int j, const int k) {
      return i + 3 * j + 9 * k;
    }

    /**
     * @brief Calculate the Cartesian/TensorProduct index from linear index
     * @param linearIndex The linear index of support point
     * @param i0 The Cartesian index in the xi0 direction
     * @param i1 The Cartesian index in the xi1 direction
     * @param i2 The Cartesian index in the xi2 direction
     */
    PROXY_HOST_DEVICE
    constexpr static void multiIndex(const int linearIndex, int &i0, int &i1, int &i2) {
      i2 = (linearIndex * 29) >> 8;
      i1 = ((linearIndex * 22) >> 6) - i2 * 3;
      i0 = linearIndex - i1 * 3 - i2 * 9;
    }

    /**
     * @brief Evaluate all basis functions at a point
     * @param coords The coordinates (in parent frame) at which to evaluate
     * @param N Array to hold the value of the basis functions
     */
    PROXY_HOST_DEVICE
    static void value(const double (&coords)[3], double (&N)[numSupportPoints]) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          for (int c = 0; c < 3; ++c) {
            const int lindex = linearIndex(a, b, c);
            N[lindex] = LagrangeBasis2_Tensorial::value(a, coords[0]) *
                        LagrangeBasis2_Tensorial::value(b, coords[1]) *
                        LagrangeBasis2_Tensorial::value(c, coords[2]);
          }
        }
      }
    }
  };
};

// Define static constexpr array
constexpr double LagrangeBasis2_Tensorial::D[3][3];

#endif /* _LAGRANGEBASIS2_TENSORIAL_HPP_ */
