#ifndef SEMQKGLINTEGRALS_HPP_
#define SEMQKGLINTEGRALS_HPP_

#include "SEMQkGLBasisFunctions.hpp"
#include "common/CArray.hpp"
#include "common/mathUtilites.hpp"

#include<stdio.h>

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
class SEMQkGLIntegrals
{
public:
  static constexpr int order = ORDER;
  static constexpr int rows = (order + 1) * (order + 1) * (order + 1);

  void init()
  {
    SEMQkGLBasisFunctions<ORDER>::gaussLobattoQuadraturePoints( m_quadraturePointCoords );
    SEMQkGLBasisFunctions<ORDER>::gaussLobattoQuadratureWeights( m_weights);
    SEMQkGLBasisFunctions<ORDER>::getDerivativeBasisFunction1D( m_quadraturePointCoords, m_dPhi);
    SEMQkGLBasisFunctions<ORDER>::getDerivativeBasisFunction1DLow( m_quadraturePointCoords, m_dPhiO1 );
  };

  // compute B and M
  template< typename VECTOR_DOUBLE_VIEW,
            typename COORDS_TYPE,
            typename ARRAY_DOUBLE_VIEW >
  SEMKERNELS_HOST_DEVICE 
  constexpr static void computeB(  VECTOR_DOUBLE_VIEW const & weights,
                                   COORDS_TYPE const & coords,
                                   ARRAY_DOUBLE_VIEW & dPhi,
                                   float massMatrixLocal[],
                                   float B[order+1][6] )
  {

    for ( int q3 = 0; q3 < order + 1; q3++ )
    {
      for ( int q2 = 0; q2 < order + 1; q2++ )
      {
        for ( int q1 = 0; q1 < order + 1; q1++ )
        {
          int const q = linearIndex( order, q1, q2, q3 );

          // compute jacobian matrix
          double jac00 = 0;
          double jac01 = 0;
          double jac02 = 0;
          double jac10 = 0;
          double jac11 = 0;
          double jac12 = 0;
          double jac20 = 0;
          double jac21 = 0;
          double jac22 = 0;

          // 1-parent direction
          for ( int a1 = 0; a1 < 2; ++a1 )
          {
            int const a = linearIndex( 1, a1, q2, q3 );
            double X = coords( a, 0 );
            double Y = coords( a, 1 );
            double Z = coords( a, 2 );

            jac00 += X * dPhi[q1][a1];
            jac10 += Y * dPhi[q1][a1];
            jac20 += Z * dPhi[q1][a1];
          }

          // 2-parent direction
          for ( int a2 = 0; a2 < 2; ++a2 )
          {
            int const a = linearIndex( 1, q1, a2, q3 );
            double X = coords( a, 0 );
            double Y = coords( a, 1 );
            double Z = coords( a, 2 );
            jac01 += X * dPhi[q2][a2];
            jac11 += Y * dPhi[q2][a2];
            jac21 += Z * dPhi[q2][a2];
          }

          // 3-parent direction
          for ( int a3 = 0; a3 < 2; ++a3 )
          {
            int const a = linearIndex( order, q1, q2, a3 );
            double X = coords( a, 0 );
            double Y = coords( a, 1 );
            double Z = coords( a, 2 );
            jac02 += X * dPhi[q3][a3];
            jac12 += Y * dPhi[q3][a3];
            jac22 += Z * dPhi[q3][a3];
          }

          // printf( "J(%2d,%2d,%2d) = | % 6.4f % 6.4f % 6.4f |\n", q1, q2, q3, jac00, jac01, jac02 );
          // printf( "              | % 6.4f % 6.4f % 6.4f |\n", jac10, jac11, jac12 );
          // printf( "              | % 6.4f % 6.4f % 6.4f |\n", jac20, jac21, jac22 );
//          printf( "\n" );

          // detJ
          double const detJ = jac00 * (jac11 * jac22 - jac21 * jac12)
                             - jac01 * (jac10 * jac22 - jac20 * jac12)
                             + jac02 * (jac10 * jac21 - jac20 * jac11);

          double const detJM1 = 1. / detJ;

          // inv of jac is equal of the minors of the transposed of jac
          double const invJac00 = ( jac11 * jac22 - jac12 * jac21 ) * detJM1;
          double const invJac01 = ( jac02 * jac21 - jac01 * jac22 ) * detJM1;
          double const invJac02 = ( jac01 * jac12 - jac02 * jac11 ) * detJM1;
          double const invJac10 = ( jac12 * jac20 - jac10 * jac22 ) * detJM1;
          double const invJac11 = ( jac00 * jac22 - jac02 * jac20 ) * detJM1;
          double const invJac12 = ( jac02 * jac10 - jac00 * jac12 ) * detJM1;
          double const invJac20 = ( jac10 * jac21 - jac11 * jac20 ) * detJM1;
          double const invJac21 = ( jac01 * jac20 - jac00 * jac21 ) * detJM1;
          double const invJac22 = ( jac00 * jac11 - jac01 * jac10 ) * detJM1;

          double const transpInvJac00 = invJac00;
          double const transpInvJac01 = invJac10;
          double const transpInvJac02 = invJac20;
          double const transpInvJac10 = invJac01;
          double const transpInvJac11 = invJac11;
          double const transpInvJac12 = invJac21;
          double const transpInvJac20 = invJac02;
          double const transpInvJac21 = invJac12;
          double const transpInvJac22 = invJac22;


          // B
          B[q][0] = (invJac00 * transpInvJac00 + invJac01 * transpInvJac10 + invJac02 * transpInvJac20); //B11
          B[q][1] = (invJac10 * transpInvJac01 + invJac11 * transpInvJac11 + invJac12 * transpInvJac21); //B22
          B[q][2] = (invJac20 * transpInvJac02 + invJac21 * transpInvJac12 + invJac22 * transpInvJac22); //B33
          B[q][3] = (invJac10 * transpInvJac02 + invJac11 * transpInvJac12 + invJac12 * transpInvJac22); //B23,B32
          B[q][4] = (invJac00 * transpInvJac02 + invJac01 * transpInvJac12 + invJac02 * transpInvJac22); //B13,B31
          B[q][5] = (invJac00 * transpInvJac01 + invJac01 * transpInvJac11 + invJac02 * transpInvJac21); //B12,B21

//          printf( "B(%2d,%2d,%2d) = | %18.14e %18.14e %18.14e %18.14e %18.14e %18.14e |\n", q1, q2, q3, B[q][0], B[q][1], B[q][2], B[q][3], B[q][4], B[q][5] );

          //M
          massMatrixLocal[q] = weights[q1] * weights[q2] * weights[q3] * detJ;
        }
      }
    }
  }

  // compute the matrix $R_{i,j}=\int_{K}{\nabla{\phi_i}.\nabla{\phi_j}dx}$
  // Marc Durufle Formulae
  template< typename VECTOR_DOUBLE_VIEW,
            typename ARRAY_DOUBLE_VIEW >
  SEMKERNELS_HOST_DEVICE 
  constexpr static void gradPhiGradPhi( const int & nPointsPerElement,
                                         VECTOR_DOUBLE_VIEW const & weights,
                                         ARRAY_DOUBLE_VIEW const & dPhi,
                                         float const B[][6],
                                         float const pnLocal[],
                                         float Y[] )
  {
    float R[rows];

    for ( int i3 = 0; i3 < ORDER + 1; i3++ )
    {
      for ( int i2 = 0; i2 < ORDER + 1; i2++ )
      {
        for ( int i1 = 0; i1 < ORDER + 1; i1++ )
        {
          for ( int j = 0; j < nPointsPerElement; j++ )
          {
            R[j] = 0;
          }

          //B11
          for ( int j1 = 0; j1 < ORDER + 1; j1++ )
          {
            int j = linearIndex( ORDER, j1, i2, i3 );
            for ( int l = 0; l < ORDER + 1; l++ )
            {
              int ll = linearIndex( ORDER, l, i2, i3 );
              R[j] += weights[l]* weights[i2] * weights[i3] * (B[ll][0] * dPhi[i1][l ]* dPhi[j1][l ]);
            }
          }
          //B22
          for ( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            int j = linearIndex( ORDER, i1, j2, i3 );
            for ( int m = 0; m < ORDER + 1; m++ )
            {
              int mm = linearIndex( ORDER, i1, m, i3 );
              R[j] += weights[i1] * weights[m]* weights[i3] * (B[mm][1] * dPhi[i2][m ]* dPhi[j2][m ]);
            }
          }
          //B33
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            int j = linearIndex( ORDER, i1, i2, j3 );
            for ( int n = 0; n < ORDER + 1; n++ )
            {
              int nn = linearIndex( ORDER, i1, i2, n );
              R[j] += weights[i1] * weights[i2] * weights[n]* (B[nn][2] * dPhi[i3][n ]* dPhi[j3][n ]);
            }
          }
          // B12,B21 (B[][3])
          for ( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            for ( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = linearIndex( ORDER, j1, j2, i3 );
              int k = linearIndex( ORDER, j1, i2, i3 );
              int l = linearIndex( ORDER, i1, j2, i3 );
              R[j] += weights[j1] * weights[i2] * weights[i3] * (B[k][3] * dPhi[i1][j1] * dPhi[j2][i2] ) +
                      weights[i1] * weights[j2] * weights[i3] * (B[l][3] * dPhi[j1][i1] * dPhi[i2][j2] );
            }
          }
          // B13,B31 (B[][4])
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for ( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = linearIndex( ORDER, j1, i2, i3 );
              int k = linearIndex( ORDER, j1, i2, i3 );
              int l = linearIndex( ORDER, j1, i2, j3 );
              R[j] += weights[j1] * weights[i2] * weights[i3] * (B[k][4] * dPhi[j1][i1] * dPhi[j3][i3] ) +
                      weights[j1] * weights[i2] * weights[j3] * (B[l][4] * dPhi[j1][i1] * dPhi[i3][j3] );
            }
          }
          // B23,B32 (B[][5])
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for ( int j2 = 0; j2 < ORDER + 1; j2++ )
            {
              int j = linearIndex( ORDER, i1, j2, j3 );
              int k = linearIndex( ORDER, i1, j2, i3 );
              int l = linearIndex( ORDER, i1, i2, j3 );
              R[j] += weights[i1] * weights[j2] * weights[i3] * (B[k][5] * dPhi[i2][i2] * dPhi[j3][i3] ) +
                      weights[i1] * weights[i2] * weights[j3] * (B[l][5] * dPhi[j2][i2] * dPhi[i3][j3] );
            }
          }

          int i = linearIndex( ORDER, i1, i2, i3 );
          Y[i] = 0;
          for ( int j = 0; j < nPointsPerElement; j++ )
          {
            Y[i] += R[j] * pnLocal[j];
          }

        }
      }
    }
  }

  // compute stiffnessVector.
  // returns mass matrix and stiffness vector local to an element
  template< typename ARRAY_DOUBLE_VIEW >
  SEMKERNELS_HOST_DEVICE 
  constexpr void computeMassMatrixAndStiffnessVector( const int & elementNumber,
                                                              const int & nPointsPerElement,
                                                              ARRAY_DOUBLE_VIEW const & nodesCoordsX,
                                                              ARRAY_DOUBLE_VIEW const & nodesCoordsY,
                                                              ARRAY_DOUBLE_VIEW const & nodesCoordsZ,
                                                              float massMatrixLocal[],
                                                              float const pnLocal[],
                                                              float Y[] )
  {

    // ***** Gather coordinates *****
    shiva::CArrayNd<double,8,3> X{ 0.0 };
    {
      int I = 0;
      for ( int k = 0; k < 2; ++k )
      {
        for ( int j = 0; j < 2; ++j )
        {
          for ( int i = 0; i < 2; ++i )
          {
            int l = linearIndex( 1, i, j, k );
            X[I][0] = nodesCoordsX( elementNumber, l );
            X[I][1] = nodesCoordsY( elementNumber, l );
            X[I][2] = nodesCoordsZ( elementNumber, l );
            ++I;
          }
        }
      }
    }


    // ***** Compute Low order Jacobian and B-matrix *****


    float B[rows][6];
    computeB( m_weights, X, m_dPhiO1, massMatrixLocal, B );

    // compute stifness  matrix ( durufle's optimization)
    gradPhiGradPhi( nPointsPerElement, m_weights, m_dPhi, B, pnLocal, Y );
  }


  
  double m_quadraturePointCoords[ORDER+1];
  double m_weights[ORDER+1];
  double m_dPhi[ORDER+1][ORDER+1];
  double m_dPhiO1[ORDER+1][2];

  /////////////////////////////////////////////////////////////////////////////////////
  //  end from first implementation
  /////////////////////////////////////////////////////////////////////////////////////

};

#endif //SEMQKGLINTEGRALS_HPP_
