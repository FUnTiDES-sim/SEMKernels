/**
 * @file Qk_Hexahedron_Tensorial.hpp
 * @brief Tensorial O(n^4) implementation of Qk hexahedral spectral elements
 *
 * This implementation uses matrix-free Kronecker product operations to achieve
 * O(n^4) complexity for stiffness operator application, compared to O(n^6) for
 * classical element-by-element assembly.
 *
 * Key features:
 * - Derivative matrix D for computing gradients
 * - Kronecker products for directional derivatives
 * - Precomputed inverse Jacobian metrics
 * - Matrix-free stiffness application via callbacks
 * - Template-based for Q1-Q5 orders
 * - API-compatible with makutu/shiva implementations
 */

#ifndef _QK_HEXAHEDRON_TENSORIAL_HPP_
#define _QK_HEXAHEDRON_TENSORIAL_HPP_

#include <data_type.h>
#include "LagrangeBasis1_Tensorial.hpp"
#include "LagrangeBasis2_Tensorial.hpp"
#include "LagrangeBasis3GL_Tensorial.hpp"
#include "LagrangeBasis4GL_Tensorial.hpp"
#include "LagrangeBasis5GL_Tensorial.hpp"
#include "common/mathUtilites.hpp"
#include "common/compile_time_loops.hpp"

/**
 * @class Qk_Hexahedron_Tensorial
 * @brief Templated hexahedral element with tensorial O(n^4) operators
 *
 * Uses the same basis functions as makutu but implements stiffness computation
 * via Kronecker products for improved performance. Compatible API with makutu/shiva.
 *
 * @tparam GL_BASIS The Lagrange basis type (LagrangeBasis*_Tensorial)
 */
template<typename GL_BASIS>
class Qk_Hexahedron_Tensorial final
{
public:
  constexpr static bool isShiva = false;
  constexpr static bool isTensorial = true;

  /// The number of nodes per element per dimension
  constexpr static int num1dNodes = GL_BASIS::numSupportPoints;

  /// Half the number of support points, rounded down. Precomputed for efficiency
  constexpr static int halfNodes = (GL_BASIS::numSupportPoints - 1) / 2;

  /// The total number of nodes per element
  constexpr static int numNodes = GL_BASIS::TensorProduct3D::numSupportPoints;

  /// The number of quadrature points per element (same as nodes for GL)
  constexpr static int numQuadraturePoints = numNodes;

  /**
   * @brief Access derivative matrix with symmetry optimization
   * @param q The index of the quadrature point
   * @param p The index of the basis function
   * @return The derivative value D[q][p]
   *
   * Exploits symmetry of Gauss-Lobatto nodes to reduce memory accesses:
   * D[q][p] = -D[n-1-q][n-1-p] for symmetric points
   */
  PROXY_HOST_DEVICE
  static inline real_t derivativeAt(const int q, const int p)
  {
    if (p <= halfNodes)
    {
      return GL_BASIS::derivativeMatrix(q, p);
    }
    else
    {
      return -GL_BASIS::derivativeMatrix(num1dNodes - 1 - q, num1dNodes - 1 - p);
    }
  }

  /// Maximum support points per element
  constexpr static int maxSupportPoints = numNodes;

  /// Number of nodes per face
  constexpr static int numNodesPerFace = num1dNodes * num1dNodes;

  struct TransformType
  {
    float data[8][3];
  };

  struct JacobianType
  {
    float data[3][3];
  };

  /**
   * @brief The linear index associated to the given 3D indices
   * @param qa The index in the first direction
   * @param qb The index in the second direction
   * @param qc The index in the third direction
   * @return The linear index in 3D
   */
  PROXY_HOST_DEVICE
  constexpr static int linearIndex3DVal(const int qa, int const qb, int const qc)
  {
    return qa + qb * num1dNodes + qc * numNodesPerFace;
  }

  /**
   * @brief Converts from mesh vertex index to linear 3D index
   * @param k The index of the mesh vertex (0-7)
   * @return The linear index in 3D
   */
  PROXY_HOST_DEVICE
  constexpr static int meshIndexToLinearIndex3D(int const k)
  {
    return linearIndex3DVal((num1dNodes - 1) * (k % 2),
                            (num1dNodes - 1) * ((k % 4) / 2),
                            (num1dNodes - 1) * (k / 4));
  }

  /**
   * @brief Helper to get interpolation coordinate from GLL point
   * @param q Index of GLL point
   * @param k Vertex index (0 or 1)
   * @return Interpolation coordinate in [0,1]
   */
  PROXY_HOST_DEVICE
  constexpr static real_t interpolationCoord(const int q, const int k)
  {
    const real_t alpha = (GL_BASIS::parentSupportCoord(q) + 1.0) / 2.0;
    return k == 0 ? (1.0 - alpha) : alpha;
  }

  /**
   * @brief Jacobian coefficient for bilinear mapping
   * @param q Quadrature point index
   * @param i Physical dimension
   * @param k Vertex component (0 or 1)
   * @param dir Direction for derivative
   * @return Coefficient for Jacobian computation
   */
  PROXY_HOST_DEVICE
  constexpr static real_t jacobianCoefficient1D(const int q, const int i, const int k, const int dir)
  {
    if (i == dir)
      return k == 0 ? -0.5 : 0.5;
    else
      return interpolationCoord(q, k);
  }

  /**
   * @brief Compute Jacobian transformation at a quadrature point
   * @param qa Quadrature index in first direction
   * @param qb Quadrature index in second direction
   * @param qc Quadrature index in third direction
   * @param X Element vertex coordinates [8][3]
   * @param J Output Jacobian matrix [3][3]
   */
  PROXY_HOST_DEVICE
  static void jacobianTransformation(int const qa,
                                     int const qb,
                                     int const qc,
                                     real_t const (&X)[8][3],
                                     real_t (&J)[3][3])
  {
    // Initialize Jacobian
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        J[i][j] = 0.0;

    // Compute Jacobian from 8 vertices using bilinear mapping
    for (int k = 0; k < 8; k++)
    {
      const int ka = k % 2;
      const int kb = (k % 4) / 2;
      const int kc = k / 4;

      for (int j = 0; j < 3; j++)
      {
        real_t jacCoeff = jacobianCoefficient1D(qa, 0, ka, j) *
                          jacobianCoefficient1D(qb, 1, kb, j) *
                          jacobianCoefficient1D(qc, 2, kc, j);
        for (int i = 0; i < 3; i++)
        {
          J[i][j] += jacCoeff * X[k][i];
        }
      }
    }
  }

  /**
   * @brief Computes the matrix B = J^{-T}J^{-1}/det(J) at a Gauss-Lobatto point
   * @param qa The 1d quadrature point index in xi0 direction
   * @param qb The 1d quadrature point index in xi1 direction
   * @param qc The 1d quadrature point index in xi2 direction
   * @param X Array containing the coordinates of the support points
   * @param J Array to store the Jacobian
   * @param B Array to store the matrix B, in Voigt notation [6]
   */
  PROXY_HOST_DEVICE
  static void computeBMatrix(int const qa,
                            int const qb,
                            int const qc,
                            real_t const (&X)[8][3],
                            real_t (&J)[3][3],
                            real_t (&B)[6])
  {
    // Compute Jacobian
    jacobianTransformation(qa, qb, qc, X, J);

    // Compute determinant
    real_t detJ = std::abs(J[0][0] * (J[1][1] * J[2][2] - J[2][1] * J[1][2])
                          - J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2])
                          + J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]));

    // Compute G = J^T J / det(J)
    real_t G[3][3];
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        G[i][j] = 0;
        for (int kk = 0; kk < 3; kk++)
        {
          G[i][j] += J[kk][i] * J[kk][j];
        }
        G[i][j] /= detJ;
      }
    }

    // Invert G to get B = det(J) * (J^T J)^{-1} in Voigt notation
    real_t det_G = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
                  - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
                  + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
    real_t inv_det = 1.0 / det_G;

    // Store in Voigt notation: B = [B_xx, B_yy, B_zz, B_yz, B_xz, B_xy]
    real_t invG[3][3];
    invG[0][0] = (G[1][1] * G[2][2] - G[1][2] * G[2][1]) * inv_det;
    invG[0][1] = (G[0][2] * G[2][1] - G[0][1] * G[2][2]) * inv_det;
    invG[0][2] = (G[0][1] * G[1][2] - G[0][2] * G[1][1]) * inv_det;
    invG[1][0] = invG[0][1];
    invG[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * inv_det;
    invG[1][2] = (G[0][2] * G[1][0] - G[0][0] * G[1][2]) * inv_det;
    invG[2][0] = invG[0][2];
    invG[2][1] = invG[1][2];
    invG[2][2] = (G[0][0] * G[1][1] - G[0][1] * G[1][0]) * inv_det;

    B[0] = invG[0][0];
    B[1] = invG[1][1];
    B[2] = invG[2][2];
    B[3] = invG[1][2];
    B[4] = invG[0][2];
    B[5] = invG[0][1];
  }

  /**
   * @brief Compute "Grad(Phi)*B*Grad(Phi)" using tensorial O(n^4) algorithm
   * @tparam qa Compile-time quadrature point index in xi0 direction
   * @tparam qb Compile-time quadrature point index in xi1 direction
   * @tparam qc Compile-time quadrature point index in xi2 direction
   * @param B Array of the B matrix, in Voigt notation
   * @param func1 Callback for quadrature point processing
   * @param func2 Callback for matrix contributions (i, j, value)
   *
   * This is the key O(n^4) implementation using derivative matrices.
   */
  template<int qa, int qb, int qc, typename FUNC1, typename FUNC2>
  PROXY_HOST_DEVICE
  static void computeGradPhiBGradPhi(real_t const (&B)[6],
                                     FUNC1 && func1,
                                     FUNC2 && func2)
  {
    // Get quadrature weight
    const real_t w = GL_BASIS::weight(qa) * GL_BASIS::weight(qb) * GL_BASIS::weight(qc);

    // Call func1 with quadrature indices
    func1(qa, qb, qc);

    constexpr int rp1 = num1dNodes;
    constexpr int rp1sq = numNodesPerFace;

    // Tensorial O(n^4) loop structure
    for (int i = 0; i < rp1; i++)
    {
      const int ibc = i + rp1 * qb + rp1sq * qc;
      const int aic = qa + rp1 * i + rp1sq * qc;
      const int abi = qa + rp1 * qb + rp1sq * i;

      // Use derivative matrix with symmetry optimization
      const real_t gia = derivativeAt(qa, i);
      const real_t gib = derivativeAt(qb, i);
      const real_t gic = derivativeAt(qc, i);

      for (int j = 0; j < rp1; j++)
      {
        const int jbc = j + rp1 * qb + rp1sq * qc;
        const int ajc = qa + rp1 * j + rp1sq * qc;
        const int abj = qa + rp1 * qb + rp1sq * j;

        const real_t gja = derivativeAt(qa, j);
        const real_t gjb = derivativeAt(qb, j);
        const real_t gjc = derivativeAt(qc, j);

        // Diagonal terms: grad_x * grad_x, grad_y * grad_y, grad_z * grad_z
        const real_t w0 = w * gia * gja;
        func2(ibc, jbc, w0 * B[0]);

        const real_t w1 = w * gib * gjb;
        func2(aic, ajc, w1 * B[1]);

        const real_t w2 = w * gic * gjc;
        func2(abi, abj, w2 * B[2]);

        // Off-diagonal terms: cross-gradient coupling
        const real_t w3 = w * gib * gjc;
        func2(aic, abj, w3 * B[3]);
        func2(abj, aic, w3 * B[3]);

        const real_t w4 = w * gia * gjc;
        func2(ibc, abj, w4 * B[4]);
        func2(abj, ibc, w4 * B[4]);

        const real_t w5 = w * gia * gjb;
        func2(ibc, ajc, w5 * B[5]);
        func2(ajc, ibc, w5 * B[5]);
      }
    }
  }

  /**
   * @brief Compute stiffness term with callbacks (API-compatible with makutu)
   * @param transformData Transform data containing vertex coordinates
   * @param func1 Callback for quadrature point processing func1(qa, qb, qc)
   * @param func2 Callback for matrix contributions func2(i, j, K_ij)
   */
  template<typename FUNC1, typename FUNC2>
  PROXY_HOST_DEVICE
  static void computeStiffnessTerm(TransformType const & transformData,
                                   FUNC1 && func1,
                                   FUNC2 && func2)
  {
    triple_loop<num1dNodes, num1dNodes, num1dNodes>([&](auto const icqa, auto const icqb, auto const icqc)
    {
      constexpr int qa = decltype(icqa)::value;
      constexpr int qb = decltype(icqb)::value;
      constexpr int qc = decltype(icqc)::value;

      real_t B[6] = {0};
      real_t J[3][3] = {{0}};
      float const (&X)[8][3] = transformData.data;

      computeBMatrix(qa, qb, qc, X, J, B);
      computeGradPhiBGradPhi<qa, qb, qc>(B, func1, func2);
    });
  }

  /**
   * @brief Compute stiffness term with Jacobian for elastic waves
   * @param transformData Transform data containing vertex coordinates
   * @param func1 Callback for quadrature point processing func1(qa, qb, qc, J)
   * @param func2 Callback for matrix contributions func2(i, j, K_ij, dir1, dir2)
   */
  template<typename FUNC1, typename FUNC2>
  PROXY_HOST_DEVICE
  static void computeStiffNessTermwithJac(TransformType const & transformData,
                                          FUNC1 && func1,
                                          FUNC2 && func2)
  {
    triple_loop<num1dNodes, num1dNodes, num1dNodes>([&](auto const icqa, auto const icqb, auto const icqc)
    {
      constexpr int qa = decltype(icqa)::value;
      constexpr int qb = decltype(icqb)::value;
      constexpr int qc = decltype(icqc)::value;
      JacobianType J = {{0}};
      jacobianTransformation(qa, qb, qc, transformData.data, J.data);
      computeGradPhiGradPhi<qa, qb, qc>(J, func1, func2);
    });
  }

  /**
   * @brief Compute gradient outer product for elastic waves (inverts Jacobian inline)
   * @tparam qa Compile-time quadrature point index in xi0 direction
   * @tparam qb Compile-time quadrature point index in xi1 direction
   * @tparam qc Compile-time quadrature point index in xi2 direction
   * @param J Jacobian matrix (will be inverted)
   * @param func1 Callback for quadrature point processing
   * @param func2 Callback for matrix contributions (i, j, value, dir1, dir2)
   */
  template<int qa, int qb, int qc, typename FUNC1, typename FUNC2>
  PROXY_HOST_DEVICE
  static void computeGradPhiGradPhi(JacobianType &J,
                                    FUNC1 && func1,
                                    FUNC2 && func2)
  {
    real_t const detJ = invert3x3(J.data);
    const real_t w = GL_BASIS::weight(qa) * GL_BASIS::weight(qb) * GL_BASIS::weight(qc);
    func1(qa, qb, qc, J.data);

    constexpr int rp1 = num1dNodes;

    // Tensorial O(n^4) loop structure
    for (int i = 0; i < rp1; i++)
    {
      const int ibc = i + rp1 * qb + rp1 * rp1 * qc;
      const int aic = qa + rp1 * i + rp1 * rp1 * qc;
      const int abi = qa + rp1 * qb + rp1 * rp1 * i;

      // Use derivative matrix with symmetry optimization
      const real_t gia = derivativeAt(qa, i);
      const real_t gib = derivativeAt(qb, i);
      const real_t gic = derivativeAt(qc, i);

      for (int j = 0; j < rp1; j++)
      {
        const int jbc = j + rp1 * qb + rp1 * rp1 * qc;
        const int ajc = qa + rp1 * j + rp1 * rp1 * qc;
        const int abj = qa + rp1 * qb + rp1 * rp1 * j;

        const real_t gja = derivativeAt(qa, j);
        const real_t gjb = derivativeAt(qb, j);
        const real_t gjc = derivativeAt(qc, j);

        // Diagonal terms
        const real_t w00 = w * gia * gja;
        func2(ibc, jbc, w00 * detJ, 0, 0);

        const real_t w11 = w * gib * gjb;
        func2(aic, ajc, w11 * detJ, 1, 1);

        const real_t w22 = w * gic * gjc;
        func2(abi, abj, w22 * detJ, 2, 2);

        // Off-diagonal terms
        const real_t w12 = w * gib * gjc;
        func2(aic, abj, w12 * detJ, 1, 2);
        func2(abj, aic, w12 * detJ, 2, 1);

        const real_t w02 = w * gia * gjc;
        func2(ibc, abj, w02 * detJ, 0, 2);
        func2(abj, ibc, w02 * detJ, 2, 0);

        const real_t w01 = w * gia * gjb;
        func2(ibc, ajc, w01 * detJ, 0, 1);
        func2(ajc, ibc, w01 * detJ, 1, 0);
      }
    }
  }

  /**
   * @brief Compute mass term with callback (API-compatible with makutu)
   * @param transformData Transform data containing vertex coordinates
   * @param func Callback function func(q, massValue)
   */
  template<typename FUNC>
  PROXY_HOST_DEVICE
  static void computeMassTerm(TransformType const & transformData, FUNC && func)
  {
    constexpr int N = num1dNodes;
    triple_loop<N, N, N>([&](auto const icqa, auto const icqb, auto const icqc)
    {
      constexpr int qa = decltype(icqa)::value;
      constexpr int qb = decltype(icqb)::value;
      constexpr int qc = decltype(icqc)::value;
      constexpr int q = GL_BASIS::TensorProduct3D::linearIndex(qa, qb, qc);
      constexpr real_t w3D = GL_BASIS::weight(qa) * GL_BASIS::weight(qb) * GL_BASIS::weight(qc);

      real_t J[3][3] = {{0}};
      float const (&X)[8][3] = transformData.data;
      jacobianTransformation(qa, qb, qc, X, J);

      real_t val = std::abs(determinant(J)) * w3D;
      func(q, val);
    });
  }

  /**
   * @brief Evaluate shape functions at a point
   * @param coords Parent coordinates [-1,1]^3
   * @param N Output shape function values
   */
  PROXY_HOST_DEVICE
  static void calcN(double const (&coords)[3], double (&N)[numNodes])
  {
    GL_BASIS::TensorProduct3D::value(coords, N);
  }

  /**
   * @brief Trilinear interpolation
   * @param alpha Interpolation coefficient in [0,1] for first coordinate
   * @param beta Interpolation coefficient in [0,1] for second coordinate
   * @param gamma Interpolation coefficient in [0,1] for third coordinate
   * @param X Real-world coordinates of cell corners
   * @param coords Real-world coordinates of interpolated point
   */
  PROXY_HOST_DEVICE
  static void trilinearInterp(real_t const alpha,
                              real_t const beta,
                              real_t const gamma,
                              real_t const (&X)[8][3],
                              real_t (&coords)[3])
  {
    for (int i = 0; i < 3; i++)
    {
      coords[i] = 0.0;
    }

    for (int k = 0; k < 8; k++)
    {
      const int ka = k % 2;
      const int kb = (k % 4) / 2;
      const int kc = k / 4;

      real_t coeff = (ka == 0 ? (1.0 - alpha) : alpha) *
                     (kb == 0 ? (1.0 - beta) : beta) *
                     (kc == 0 ? (1.0 - gamma) : gamma);

      for (int i = 0; i < 3; i++)
      {
        coords[i] += coeff * X[k][i];
      }
    }
  }

  /**
   * @brief Virtual functions for compatibility
   */
  PROXY_HOST_DEVICE
  virtual int getNumQuadraturePoints() { return numQuadraturePoints; }

  PROXY_HOST_DEVICE
  virtual int getNumSupportPoints() { return numNodes; }

  PROXY_HOST_DEVICE
  virtual int getMaxSupportPoints() const { return maxSupportPoints; }
};

// Type aliases for each order
using Q1_Hexahedron_Lagrange_GaussLobatto_Tensorial = Qk_Hexahedron_Tensorial<LagrangeBasis1_Tensorial>;
using Q2_Hexahedron_Lagrange_GaussLobatto_Tensorial = Qk_Hexahedron_Tensorial<LagrangeBasis2_Tensorial>;
using Q3_Hexahedron_Lagrange_GaussLobatto_Tensorial = Qk_Hexahedron_Tensorial<LagrangeBasis3GL_Tensorial>;
using Q4_Hexahedron_Lagrange_GaussLobatto_Tensorial = Qk_Hexahedron_Tensorial<LagrangeBasis4GL_Tensorial>;
using Q5_Hexahedron_Lagrange_GaussLobatto_Tensorial = Qk_Hexahedron_Tensorial<LagrangeBasis5GL_Tensorial>;

/**
 * @brief Selector template for compile-time order selection
 */
template<int ORDER>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector;

template<>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector<1>
{
  using type = Q1_Hexahedron_Lagrange_GaussLobatto_Tensorial;
};

template<>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector<2>
{
  using type = Q2_Hexahedron_Lagrange_GaussLobatto_Tensorial;
};

template<>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector<3>
{
  using type = Q3_Hexahedron_Lagrange_GaussLobatto_Tensorial;
};

template<>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector<4>
{
  using type = Q4_Hexahedron_Lagrange_GaussLobatto_Tensorial;
};

template<>
struct Qk_Hexahedron_Lagrange_GaussLobatto_Tensorial_Selector<5>
{
  using type = Q5_Hexahedron_Lagrange_GaussLobatto_Tensorial;
};

#endif /* _QK_HEXAHEDRON_TENSORIAL_HPP_ */
