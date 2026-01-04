#ifndef _LAGRANGEBASIS4GL_TENSORIAL_HPP_
#define _LAGRANGEBASIS4GL_TENSORIAL_HPP_

/**
 * @file LagrangeBasis4GL_Tensorial.hpp
 * @brief Tensorial implementation of Q4 Lagrange basis with Gauss-Lobatto points
 *
 * This implements an O(n^4) matrix-free stiffness operator using:
 * - Derivative matrices (D matrix)
 * - Kronecker product operations
 * - Precomputed geometric metrics
 */

#include <data_type.h>

/**
 * @class LagrangeBasis4GL_Tensorial
 * @brief Fourth-order Lagrange polynomial basis with Gauss-Lobatto quadrature points
 *
 * Parent space defined by:
 *                 o------o------o------o------o  ---> xi
 *  Index:         0      1      2      3      4
 *  Coordinate:   -1 -sqrt(3/7)  0  sqrt(3/7)  1
 */
class LagrangeBasis4GL_Tensorial {
public:
  /// The number of support points for the basis
  constexpr static int numSupportPoints = 5;

  /// sqrt(3/7)
  constexpr static double sqrt3_7 = 0.6546536707079771;

  /**
   * @brief The value of the weight for the given support point
   * @param q The index of the support point
   * @return The value of the quadrature weight
   */
  PROXY_HOST_DEVICE
  constexpr static double weight(const int q) {
    switch (q) {
    case 0:
    case 4:
      return 1.0 / 10.0;
    case 1:
    case 3:
      return 49.0 / 90.0;
    default:
      return 32.0 / 45.0;
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
      result = -sqrt3_7;
      break;
    case 2:
      result = 0.0;
      break;
    case 3:
      result = sqrt3_7;
      break;
    case 4:
      result = 1.0;
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @brief Gradient of basis function evaluated at quadrature point
   * @param q The index of the quadrature point
   * @param p The index of the basis function (assumed in 0..(N-1)/2 by symmetry)
   * @return d(phi_p)/d(xi) evaluated at xi_q
   */
  PROXY_HOST_DEVICE
  constexpr static double gradientAt(const int q, const int p) {
    switch (q) {
    case 0:
      switch (p) {
      case 0: return -5.0000000000000000000;
      case 1: return -1.2409902530309828578;
      case 2: return 0.37500000000000000000;
      }
      break;
    case 1:
      switch (p) {
      case 0: return 6.7565024887242400038;
      case 1: return 0.0;
      case 2: return -1.3365845776954533353;
      }
      break;
    case 2:
      switch (p) {
      case 0: return -2.6666666666666666667;
      case 1: return 1.7457431218879390501;
      case 2: return 0.0;
      }
      break;
    case 3:
      switch (p) {
      case 0: return 1.4101641779424266628;
      case 1: return -0.7637626158259733344;
      case 2: return 1.3365845776954533353;
      }
      break;
    case 4:
      switch (p) {
      case 0: return -0.50000000000000000000;
      case 1: return 0.25900974696901714215;
      case 2: return -0.37500000000000000000;
      }
      break;
    }
    return 0;
  }

  /**
   * @brief Access element (i,j) of the derivative matrix D
   * @param i Row index
   * @param j Column index
   * @return Value of D[i][j]
   *
   * Uses constexpr local array for compile-time evaluation and device compatibility
   */
  PROXY_HOST_DEVICE
  constexpr static double derivativeMatrix(const int i, const int j) {
    constexpr double D[5][5] = {
        {-5.0, 6.7565024887242400038, -2.6666666666666666667, 1.4101641779424266628, -0.5},
        {-1.2409902530309828578, 0.0, 1.7457431218879390501, -0.7637626158259733344, 0.25900974696901714215},
        {0.37500000000000000000, -1.3365845776954533353, 0.0, 1.3365845776954533353, -0.37500000000000000000},
        {-0.25900974696901714215, 0.7637626158259733344, -1.7457431218879390501, 0.0, 1.2409902530309828578},
        {0.5, -1.4101641779424266628, 2.6666666666666666667, -6.7565024887242400038, 5.0}
    };
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
    double result = 0.0;

    switch (index) {
    case 0:
      result = (1.0 / 8.0) * (-1.0 + xi) * xi * (-3.0 + 7.0 * xi * xi);
      break;
    case 1:
      result = (49.0 / 24.0) * (sqrt3_7 - xi) * xi * (-1.0 + xi * xi);
      break;
    case 2:
      result = (1.0 / 3.0) * (3.0 - 10.0 * xi * xi + 7.0 * xi * xi * xi * xi);
      break;
    case 3:
      result = -(49.0 / 24.0) * (sqrt3_7 + xi) * xi * (-1.0 + xi * xi);
      break;
    case 4:
      result = (1.0 / 8.0) * (1.0 + xi) * xi * (-3.0 + 7.0 * xi * xi);
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
      result = (1.0 / 8.0) * (3.0 + xi * (-6.0 + 7.0 * xi * (-3.0 + 4.0 * xi)));
      break;
    case 1:
      result = (49.0 / 24.0) * (-sqrt3_7 + xi * (2.0 + 3.0 * sqrt3_7 * xi - 4.0 * xi * xi));
      break;
    case 2:
      result = (4.0 / 3.0) * xi * (-5.0 + 7.0 * xi * xi);
      break;
    case 3:
      result = (49.0 / 24.0) * (sqrt3_7 + xi * (2.0 - 3.0 * sqrt3_7 * xi - 4.0 * xi * xi));
      break;
    case 4:
      result = (1.0 / 8.0) * (-3.0 + xi * (-6.0 + 7.0 * xi * (3.0 + 4.0 * xi)));
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @class TensorProduct3D
   * @brief 3D tensor product space for Q4 elements
   */
  struct TensorProduct3D {
    /// The number of support points in the 3D tensor product (5^3 = 125)
    constexpr static int numSupportPoints = 125;

    /**
     * @brief Calculates the linear index from ijk coordinates
     * @param i The index in the xi0 direction (0-4)
     * @param j The index in the xi1 direction (0-4)
     * @param k The index in the xi2 direction (0-4)
     * @return The linear index (0-124)
     */
    PROXY_HOST_DEVICE
    constexpr static int linearIndex(const int i, const int j, const int k) {
      return i + 5 * j + 25 * k;
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
      i2 = linearIndex / 25;
      i1 = (linearIndex % 25) / 5;
      i0 = (linearIndex % 25) % 5;
    }

    /**
     * @brief Evaluate all basis functions at a point
     * @param coords The coordinates (in parent frame) at which to evaluate
     * @param N Array to hold the value of the basis functions
     */
    PROXY_HOST_DEVICE
    static void value(const double (&coords)[3], double (&N)[numSupportPoints]) {
      for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) {
          for (int c = 0; c < 5; ++c) {
            const int lindex = linearIndex(a, b, c);
            N[lindex] = LagrangeBasis4GL_Tensorial::value(a, coords[0]) *
                        LagrangeBasis4GL_Tensorial::value(b, coords[1]) *
                        LagrangeBasis4GL_Tensorial::value(c, coords[2]);
          }
        }
      }
    }
  };
};

// Define static constexpr array

#endif /* _LAGRANGEBASIS4GL_TENSORIAL_HPP_ */
