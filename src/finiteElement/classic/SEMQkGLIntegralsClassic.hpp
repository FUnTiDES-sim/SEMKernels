#ifndef SEMQKGLINTEGRALSCLASSIC_HPP_
#define SEMQKGLINTEGRALSCLASSIC_HPP_

#include "dataType.hpp"
#include "SEMmacros.hpp"
#include <fe/SEMKernels/src/finiteElement/classic/SEMQkGLBasisFunctionsClassic.hpp>
using namespace std;

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
class SEMQkGLIntegralsClassic
{
private:
  SEMQkGLBasisFunctionsClassic GLBasis;

public:
  constexpr static bool isClassic = true;

  struct PrecomputedData
  {
    float quadraturePoints[ ORDER + 1 ];
    float weights[ ORDER + 1 ];
    float derivativeBasisFunction1D[ORDER + 1][ ORDER + 1 ];
  };


  PROXY_HOST_DEVICE
  static void init( PrecomputedData & precomputedData )
  {
    // initialize quadrature points and weights
    SEMQkGLBasisFunctionsClassic::gaussLobattoQuadraturePoints( ORDER, precomputedData.quadraturePoints );
    SEMQkGLBasisFunctionsClassic::gaussLobattoQuadratureWeights( ORDER, precomputedData.weights );

    // initialize derivative basis function
    SEMQkGLBasisFunctionsClassic::getDerivativeBasisFunction1D( ORDER,
                                                                precomputedData.quadraturePoints,
                                                                precomputedData.derivativeBasisFunction1D );
  }

  PROXY_HOST_DEVICE SEMQkGLIntegralsClassic(){};
  PROXY_HOST_DEVICE ~SEMQkGLIntegralsClassic(){};

  // compute B and M
  PROXY_HOST_DEVICE
  void static
  computeB( const int & elementNumber,
            float const (&weights)[ORDER + 1],
            const float (*nodesCoords)[3],
            float const (&dPhi)[ORDER + 1][ORDER + 1],
            float massMatrixLocal[],
            float B[][COL] )
  {
    for( int i3 = 0; i3 < ORDER + 1; i3++ )
    {
      for( int i2 = 0; i2 < ORDER + 1; i2++ )
      {
        for( int i1 = 0; i1 < ORDER + 1; i1++ )
        {
          int i = i1 + i2 * (ORDER + 1) + i3 * (ORDER + 1) * (ORDER + 1);
          // compute jacobian matrix
          float jac00 = 0;
          float jac01 = 0;
          float jac02 = 0;
          float jac10 = 0;
          float jac11 = 0;
          float jac12 = 0;
          float jac20 = 0;
          float jac21 = 0;
          float jac22 = 0;

          for( int j1 = 0; j1 < ORDER + 1; j1++ )
          {
            int j = j1 + i2 * (ORDER + 1) + i3 * (ORDER + 1) * (ORDER + 1);
            float X = nodesCoords[j][0];
            float Y = nodesCoords[j][1];
            float Z = nodesCoords[j][2];
            jac00 += X * dPhi[ i1 ][ j1 ];
            jac20 += Y * dPhi[ i1 ][ j1 ];
            jac10 += Z * dPhi[ i1 ][ j1 ];
          }
          for( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            int j = i1 + j2 * (ORDER + 1) + i3 * (ORDER + 1) * (ORDER + 1);
            float X = nodesCoords[j][0];
            float Y = nodesCoords[j][1];
            float Z = nodesCoords[j][2];
            jac01 += X * dPhi[ i2 ][ j2 ];
            jac21 += Y * dPhi[ i2 ][ j2 ];
            jac11 += Z * dPhi[ i2 ][ j2 ];
          }
          for( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            int j = i1 + i2 * (ORDER + 1) + j3 * (ORDER + 1) * (ORDER + 1);
            float X = nodesCoords[j][0];
            float Y = nodesCoords[j][1];
            float Z = nodesCoords[j][2];
            jac02 += X * dPhi[ i3 ][ j3 ];
            jac22 += Y * dPhi[ i3 ][ j3 ];
            jac12 += Z * dPhi[ i3 ][ j3 ];
          }
          // detJ
          float detJ = abs( jac00 * (jac11 * jac22 - jac21 * jac12)
                            - jac01 * (jac10 * jac22 - jac20 * jac12)
                            + jac02 * (jac10 * jac21 - jac20 * jac11));

          // inv of jac is equal of the minors of the transposed of jac
          float invJac00 = jac11 * jac22 - jac12 * jac21;
          float invJac01 = jac02 * jac21 - jac01 * jac22;
          float invJac02 = jac01 * jac12 - jac02 * jac11;
          float invJac10 = jac12 * jac20 - jac10 * jac22;
          float invJac11 = jac00 * jac22 - jac02 * jac20;
          float invJac12 = jac02 * jac10 - jac00 * jac12;
          float invJac20 = jac10 * jac21 - jac11 * jac20;
          float invJac21 = jac01 * jac20 - jac00 * jac21;
          float invJac22 = jac00 * jac11 - jac01 * jac10;

          float transpInvJac00 = invJac00;
          float transpInvJac01 = invJac10;
          float transpInvJac02 = invJac20;
          float transpInvJac10 = invJac01;
          float transpInvJac11 = invJac11;
          float transpInvJac12 = invJac21;
          float transpInvJac20 = invJac02;
          float transpInvJac21 = invJac12;
          float transpInvJac22 = invJac22;

          float detJM1 = 1. / detJ;

          // B
          B[i][0] = (invJac00 * transpInvJac00 + invJac01 * transpInvJac10 + invJac02 * transpInvJac20) * detJM1; //B11
          B[i][1] = (invJac10 * transpInvJac01 + invJac11 * transpInvJac11 + invJac12 * transpInvJac21) * detJM1; //B22
          B[i][2] = (invJac20 * transpInvJac02 + invJac21 * transpInvJac12 + invJac22 * transpInvJac22) * detJM1; //B33
          B[i][3] = (invJac00 * transpInvJac01 + invJac01 * transpInvJac11 + invJac02 * transpInvJac21) * detJM1; //B12,B21
          B[i][4] = (invJac00 * transpInvJac02 + invJac01 * transpInvJac12 + invJac02 * transpInvJac22) * detJM1; //B13,B31
          B[i][5] = (invJac10 * transpInvJac02 + invJac11 * transpInvJac12 + invJac12 * transpInvJac22) * detJM1; //B23,B32

          //M
          massMatrixLocal[i] = weights[i1] * weights[i2] * weights[i3] * detJ;
        }
      }
    }
  }

  // compute the matrix $R_{i,j}=\int_{K}{\nabla{\phi_i}.\nabla{\phi_j}dx}$
  // Marc Durufle Formulae
  PROXY_HOST_DEVICE
  static void gradPhiGradPhi( const int & nPointsPerElement,
                              float const (&weights)[ORDER + 1],
                              float const (&dPhi)[ORDER + 1][ORDER + 1],
                              float const B[][COL],
                              float const pnLocal[],
                              float R[],
                              float Y[] )
  {
    constexpr int orderPow2 = (ORDER + 1) * (ORDER + 1);
    for( int i3 = 0; i3 < ORDER + 1; i3++ )
    {
      for( int i2 = 0; i2 < ORDER + 1; i2++ )
      {
        for( int i1 = 0; i1 < ORDER + 1; i1++ )
        {
          for( int j = 0; j < nPointsPerElement; j++ )
          {
            R[j] = 0;
          }

          //B11
          for( int j1 = 0; j1 < ORDER + 1; j1++ )
          {
            int j = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
            for( int l = 0; l < ORDER + 1; l++ )
            {
              int ll = l + i2 * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[l] * weights[i2] * weights[i3] * (B[ll][0] * dPhi[ l ][ i1 ] * dPhi[ l ][ j1 ]);
            }
          }
          //B22
          for( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            int j = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
            for( int m = 0; m < ORDER + 1; m++ )
            {
              int mm = i1 + m * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[i1] * weights[m] * weights[i3] * (B[mm][1] * dPhi[ m ][ i2 ] * dPhi[ m ][ j2 ]);
            }
          }
          //B33
          for( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            int j = i1 + i2 * (ORDER + 1) + j3 * orderPow2;
            for( int n = 0; n < ORDER + 1; n++ )
            {
              int nn = i1 + i2 * (ORDER + 1) + n * orderPow2;
              R[j] += weights[i1] * weights[i2] * weights[n] * (B[nn][2] * dPhi[ n ][ i3 ] * dPhi[ n ][ j3 ]);
            }
          }
          // B12,B21 (B[][3])
          for( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            for( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = j1 + j2 * (ORDER + 1) + i3 * orderPow2;
              int k = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int l = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[j1] * weights[i2] * weights[i3] * ( B[k][3] * dPhi[ j1 ][ i1 ] * dPhi[ i2 ][ j2 ] ) +
                      weights[i1] * weights[j2] * weights[i3] * ( B[l][3] * dPhi[ i1 ][ j1 ] * dPhi[ j2 ][ i2 ] );
            }
          }
          // B13,B31 (B[][4])
          for( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int k = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int l = j1 + i2 * (ORDER + 1) + j3 * orderPow2;
              R[j] += weights[j1] * weights[i2] * weights[i3] * ( B[k][4] * dPhi[ i1 ][ j1 ] * dPhi[ i3 ][ j3 ] ) +
                      weights[j1] * weights[i2] * weights[j3] * ( B[l][4] * dPhi[ i1 ][ j1 ] * dPhi[ j3 ][ i3 ] );
            }
          }
          // B23,B32 (B[][5])
          for( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for( int j2 = 0; j2 < ORDER + 1; j2++ )
            {
              int j = i1 + j2 * (ORDER + 1) + j3 * orderPow2;
              int k = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
              int l = i1 + i2 * (ORDER + 1) + j3 * orderPow2;
              R[j] += weights[i1] * weights[j2] * weights[i3] * (B[k][5] * dPhi[ i2 ][ i2 ] * dPhi[ i3 ][ j3 ]) +
                      weights[i1] * weights[i2] * weights[j3] * (B[l][5] * dPhi[ i2 ][ j2 ] * dPhi[ j3 ][ i3 ]);
            }
          }

          int i = i1 + i2 * (ORDER + 1) + i3 * orderPow2;
          Y[i] = 0;
          for( int j = 0; j < nPointsPerElement; j++ )
          {
            Y[i] += R[j] * pnLocal[j];
          }

        }
      }
    }
  }

  // compute stiffnessVector.
  // returns mass matrix and stiffness vector local to an element
  PROXY_HOST_DEVICE
  static void computeMassMatrixAndStiffnessVector( const int & elementNumber,
                                                   const int & nPointsPerElement,
                                                   float const cornersCoords[8][3],
                                                   PrecomputedData const & precomputedData,
                                                   float massMatrixLocal[],
                                                   float const pnLocal[],
                                                   float Y[] )
  {
    float B[ROW][COL];
    float R[ROW];
    // interpolate GLL nodes
    constexpr int total_points = getNumGLLPoints();
    float nodesCoords[total_points][3];
    generateElementCoordinates( cornersCoords, nodesCoords );
    // compute Jacobian, massMatrix and B
    computeB( elementNumber,
              precomputedData.weights,
              nodesCoords,
              precomputedData.derivativeBasisFunction1D,
              massMatrixLocal,
              B );

    // compute stifness  matrix ( durufle's optimization)
    gradPhiGradPhi( nPointsPerElement,
                    precomputedData.weights,
                    precomputedData.derivativeBasisFunction1D,
                    B,
                    pnLocal,
                    R,
                    Y );
  }

  static constexpr int getNumGLLPoints()
  {
    return (ORDER + 1) * (ORDER + 1) * (ORDER + 1);
  }


  // Function takes [8][3] input and [gllpoints][3] output
  PROXY_HOST_DEVICE
  static constexpr void generateElementCoordinates( const float corners[8][3], float points[][3] )
  {
    static_assert( ORDER >= 1 && ORDER <= 5, "ORDER must be between 1 and 5" );
    constexpr int n = ORDER + 1;

    constexpr float sqrt5 = 2.2360679774997897f;
    constexpr float gll_1d[6][6] = {
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},    // ORDER 0 (unused)
      {-1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},    // ORDER 1
      {-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},    // ORDER 2
      {-1.0f, -1.0f / sqrt5, 1.0f / sqrt5, 1.0f, 0.0f, 0.0f}, // ORDER 3
      {-1.0f, -0.654653670707977f, 0.0f, 0.654653670707977f, 1.0f, 0.0f},    // ORDER 4
      {-1.0f, -0.765055323929465f, -0.285231516480645f, 0.285231516480645f, 0.765055323929465f, 1.0f}    // ORDER 5
    };

    constexpr int mapping[8] = {0, 1, 3, 2, 4, 5, 7, 6};

    float reordered_corners[8][3];
    for( int i = 0; i < 8; i++ )
    {
      for( int coord = 0; coord < 3; coord++ )
      {
        reordered_corners[i][coord] = corners[mapping[i]][coord];
      }
    }

    int point_idx = 0;
    for ( int i3 = 0; i3 < n; i3++ )
    {
      for ( int i2 = 0; i2 < n; i2++ )
      {
        for ( int i1 = 0; i1 < n; i1++ )
        {
          float xi = gll_1d[ORDER][i1];
          float eta = gll_1d[ORDER][i2];
          float zeta = gll_1d[ORDER][i3];

          float u = xi;
          float v = eta;
          float w = zeta;

          for ( int coord = 0; coord < 3; coord++ )
          {
            points[point_idx][coord] =
              reordered_corners[0][coord] * (1 - u) * (1 - v) * (1 - w) * 0.125f +  // (-1,-1,-1)
              reordered_corners[1][coord] * (1 + u) * (1 - v) * (1 - w) * 0.125f +  // (+1,-1,-1)
              reordered_corners[2][coord] * (1 + u) * (1 + v) * (1 - w) * 0.125f +  // (+1,+1,-1)
              reordered_corners[3][coord] * (1 - u) * (1 + v) * (1 - w) * 0.125f +  // (-1,+1,-1)
              reordered_corners[4][coord] * (1 - u) * (1 - v) * (1 + w) * 0.125f +  // (-1,-1,+1)
              reordered_corners[5][coord] * (1 + u) * (1 - v) * (1 + w) * 0.125f +  // (+1,-1,+1)
              reordered_corners[6][coord] * (1 + u) * (1 + v) * (1 + w) * 0.125f +  // (+1,+1,+1)
              reordered_corners[7][coord] * (1 - u) * (1 + v) * (1 + w) * 0.125f;   // (-1,+1,+1)
          }
          point_idx++;
        }
      }
    }
  }
};

#endif //SEMQKGLINTEGRALSCLASSIC_HPP_
