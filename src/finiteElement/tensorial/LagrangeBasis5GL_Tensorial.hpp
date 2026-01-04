#ifndef _LAGRANGEBASIS5GL_TENSORIAL_HPP_
#define _LAGRANGEBASIS5GL_TENSORIAL_HPP_

/**
 * @file LagrangeBasis5GL_Tensorial.hpp
 * @brief Tensorial implementation of Q5 Lagrange basis with Gauss-Lobatto points
 *
 * This implements an O(n^4) matrix-free stiffness operator using:
 * - Derivative matrices (D matrix)
 * - Kronecker product operations
 * - Precomputed geometric metrics
 */

#include <data_type.h>

/**
 * @class LagrangeBasis5GL_Tensorial
 * @brief Fifth-order Lagrange polynomial basis with Gauss-Lobatto quadrature points
 *
 * Parent space defined by:
 *                 o------o------o------o------o------o  ---> xi
 *  Index:         0      1      2      3      4      5
 *  Coordinate:   -1    -a1     -a2    +a2    +a1     1
 *  where a1 = sqrt(1/21(7+2*sqrt(7))), a2 = sqrt(1/21(7-2*sqrt(7)))
 */
class LagrangeBasis5GL_Tensorial {
public:
  /// The number of support points for the basis
  constexpr static int numSupportPoints = 6;

  /// sqrt(7)
  static constexpr double sqrt_7_ = 2.64575131106459059;

  /// sqrt(7 + 2 * sqrt(7))
  static constexpr double sqrt__7_plus_2sqrt7__ = 3.50592393273573196;

  /// sqrt(7 - 2 * sqrt(7))
  static constexpr double sqrt__7_mins_2sqrt7__ = 1.30709501485960033;

  /// sqrt(1/21)
  static constexpr double sqrt_inv21 = 0.218217890235992381;

  /**
   * @brief The value of the weight for the given support point
   * @param q The index of the support point
   * @return The value of the quadrature weight
   */
  PROXY_HOST_DEVICE
  constexpr static double weight(const int q) {
    switch (q) {
    case 1:
    case 4:
      return (1.0 / 30.0) * (14.0 - sqrt_7_);
    case 2:
    case 3:
      return (1.0 / 30.0) * (14.0 + sqrt_7_);
    default:
      return 1.0 / 15.0;
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
      result = -sqrt_inv21 * sqrt__7_plus_2sqrt7__;
      break;
    case 2:
      result = -sqrt_inv21 * sqrt__7_mins_2sqrt7__;
      break;
    case 3:
      result = sqrt_inv21 * sqrt__7_mins_2sqrt7__;
      break;
    case 4:
      result = sqrt_inv21 * sqrt__7_plus_2sqrt7__;
      break;
    case 5:
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
      case 0: return -7.5000000000000000000;
      case 1: return -1.7863649483390948939;
      case 2: return 0.48495104785356916930;
      }
      break;
    case 1:
      switch (p) {
      case 0: return 10.14141593631966928023;
      case 1: return 0.0;
      case 2: return -1.72125695283023338321;
      }
      break;
    case 2:
      switch (p) {
      case 0: return -4.03618727030534800527;
      case 1: return 2.5234267774294554319088;
      case 2: return 0.0;
      }
      break;
    case 3:
      switch (p) {
      case 0: return 2.2446846481761668242712;
      case 1: return -1.1528281585359293413318;
      case 2: return 1.7529619663678659788775;
      }
      break;
    case 4:
      switch (p) {
      case 0: return -1.3499133141904880992312;
      case 1: return 0.6535475074298001672007;
      case 2: return -0.7863566722232407374395;
      }
      break;
    case 5:
      switch (p) {
      case 0: return 0.500000000000000000000;
      case 1: return -0.2377811779842313638052;
      case 2: return 0.2697006108320389724720;
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
    constexpr double D[6][6] = {
        {-7.5, 10.14141593631966928023, -4.03618727030534800527, 2.2446846481761668242712, -1.3499133141904880992312, 0.5},
        {-1.7863649483390948939, 0.0, 2.5234267774294554319088, -1.1528281585359293413318, 0.6535475074298001672007, -0.2377811779842313638052},
        {0.48495104785356916930, -1.72125695283023338321, 0.0, 1.7529619663678659788775, -0.7863566722232407374395, 0.2697006108320389724720},
        {-0.2697006108320389724720, 0.7863566722232407374395, -1.7529619663678659788775, 0.0, 1.72125695283023338321, -0.48495104785356916930},
        {0.2377811779842313638052, -0.6535475074298001672007, 1.1528281585359293413318, -2.5234267774294554319088, 0.0, 1.7863649483390948939},
        {-0.5, 1.3499133141904880992312, -2.2446846481761668242712, 4.03618727030534800527, -10.14141593631966928023, 7.5}
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
    double lambda3 = sqrt_inv21 * sqrt__7_mins_2sqrt7__;
    double lambda4 = sqrt_inv21 * sqrt__7_plus_2sqrt7__;

    switch (index) {
    case 0:
      result = (-21.0 / 16.0) * (xi * xi * xi * xi * xi - xi * xi * xi * xi -
                                 (lambda3 * lambda3 + lambda4 * lambda4) * xi * xi * xi +
                                 (lambda3 * lambda3 + lambda4 * lambda4) * xi * xi +
                                 lambda3 * lambda3 * lambda4 * lambda4 * xi -
                                 lambda3 * lambda3 * lambda4 * lambda4);
      break;
    case 1:
      result = ((21.0 / 16.0) * 2.382671682055189) *
               (xi * xi * xi * xi * xi - lambda4 * xi * xi * xi * xi -
                (lambda3 * lambda3 + 1) * xi * xi * xi +
                lambda4 * (lambda3 * lambda3 + 1) * xi * xi +
                lambda3 * lambda3 * xi - lambda4 * lambda3 * lambda3);
      break;
    case 2:
      result = ((-21.0 / 16.0) * 2.884939454396278) *
               (xi * xi * xi * xi * xi - lambda3 * xi * xi * xi * xi -
                (lambda4 * lambda4 + 1) * xi * xi * xi +
                lambda3 * (lambda4 * lambda4 + 1) * xi * xi +
                lambda4 * lambda4 * xi - lambda3 * lambda4 * lambda4);
      break;
    case 3:
      result = ((21.0 / 16.0) * 2.884939454396278) *
               (xi * xi * xi * xi * xi + lambda3 * xi * xi * xi * xi -
                (lambda4 * lambda4 + 1) * xi * xi * xi -
                lambda3 * (lambda4 * lambda4 + 1) * xi * xi +
                lambda4 * lambda4 * xi + lambda3 * lambda4 * lambda4);
      break;
    case 4:
      result = ((-21.0 / 16.0) * 2.382671682055189) *
               (xi * xi * xi * xi * xi + lambda4 * xi * xi * xi * xi -
                (lambda3 * lambda3 + 1) * xi * xi * xi -
                lambda4 * (lambda3 * lambda3 + 1) * xi * xi +
                lambda3 * lambda3 * xi + lambda4 * lambda3 * lambda3);
      break;
    case 5:
      result = (21.0 / 16.0) * (xi * xi * xi * xi * xi + xi * xi * xi * xi -
                                (lambda4 * lambda4 + lambda3 * lambda3) * xi * xi * xi -
                                (lambda3 * lambda3 + lambda4 * lambda4) * xi * xi +
                                lambda3 * lambda3 * lambda4 * lambda4 * xi +
                                lambda4 * lambda4 * lambda3 * lambda3);
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
    double lambda3 = sqrt_inv21 * sqrt__7_mins_2sqrt7__;
    double lambda4 = sqrt_inv21 * sqrt__7_plus_2sqrt7__;

    switch (index) {
    case 0:
      result = (-21.0 / 16.0) * (5.0 * xi * xi * xi * xi - 4.0 * xi * xi * xi -
                                 3.0 * (lambda3 * lambda3 + lambda4 * lambda4) * xi * xi +
                                 2.0 * (lambda3 * lambda3 + lambda4 * lambda4) * xi +
                                 lambda3 * lambda3 * lambda4 * lambda4);
      break;
    case 1:
      result = (21.0 / 16.0) * 2.382671682055189 *
               (5.0 * xi * xi * xi * xi - 4.0 * lambda4 * xi * xi * xi -
                3.0 * (lambda3 * lambda3 + 1.0) * xi * xi +
                2.0 * lambda4 * (lambda3 * lambda3 + 1.0) * xi + lambda3 * lambda3);
      break;
    case 2:
      result = (-21.0 / 16.0) * 2.884939454396278 *
               (5.0 * xi * xi * xi * xi - 4.0 * lambda3 * xi * xi * xi -
                3.0 * (lambda4 * lambda4 + 1.0) * xi * xi +
                2.0 * lambda3 * (lambda4 * lambda4 + 1.0) * xi + lambda4 * lambda4);
      break;
    case 3:
      result = (21.0 / 16.0) * 2.884939454396278 *
               (5.0 * xi * xi * xi * xi + 4.0 * lambda3 * xi * xi * xi -
                3.0 * (lambda4 * lambda4 + 1.0) * xi * xi -
                2.0 * lambda3 * (lambda4 * lambda4 + 1.0) * xi + lambda4 * lambda4);
      break;
    case 4:
      result = (-21.0 / 16.0) * 2.382671682055189 *
               (5.0 * xi * xi * xi * xi + 4.0 * lambda4 * xi * xi * xi -
                3.0 * (lambda3 * lambda3 + 1.0) * xi * xi -
                2.0 * lambda4 * (lambda3 * lambda3 + 1.0) * xi + lambda3 * lambda3);
      break;
    case 5:
      result = (21.0 / 16.0) * (5.0 * xi * xi * xi * xi + 4.0 * xi * xi * xi -
                                3.0 * (lambda3 * lambda3 + lambda4 * lambda4) * xi * xi -
                                2.0 * (lambda3 * lambda3 + lambda4 * lambda4) * xi +
                                lambda3 * lambda3 * lambda4 * lambda4);
      break;
    default:
      break;
    }

    return result;
  }

  /**
   * @class TensorProduct3D
   * @brief 3D tensor product space for Q5 elements
   */
  struct TensorProduct3D {
    /// The number of support points in the 3D tensor product (6^3 = 216)
    constexpr static int numSupportPoints = 216;

    /**
     * @brief Calculates the linear index from ijk coordinates
     * @param i The index in the xi0 direction (0-5)
     * @param j The index in the xi1 direction (0-5)
     * @param k The index in the xi2 direction (0-5)
     * @return The linear index (0-215)
     */
    PROXY_HOST_DEVICE
    constexpr static int linearIndex(const int i, const int j, const int k) {
      return i + 6 * j + 36 * k;
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
      i2 = linearIndex / 36;
      i1 = (linearIndex % 36) / 6;
      i0 = (linearIndex % 36) % 6;
    }

    /**
     * @brief Evaluate all basis functions at a point
     * @param coords The coordinates (in parent frame) at which to evaluate
     * @param N Array to hold the value of the basis functions
     */
    PROXY_HOST_DEVICE
    static void value(const double (&coords)[3], double (&N)[numSupportPoints]) {
      for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
          for (int c = 0; c < 6; ++c) {
            const int lindex = linearIndex(a, b, c);
            N[lindex] = LagrangeBasis5GL_Tensorial::value(a, coords[0]) *
                        LagrangeBasis5GL_Tensorial::value(b, coords[1]) *
                        LagrangeBasis5GL_Tensorial::value(c, coords[2]);
          }
        }
      }
    }
  };
};

// Define static constexpr array

#endif /* _LAGRANGEBASIS5GL_TENSORIAL_HPP_ */
