#ifndef SEMQKGLINTEGRALSCLASSIC_HPP_
#define SEMQKGLINTEGRALSCLASSIC_HPP_

//#include "dataType.hpp"
//#include "SEMmacros.hpp"
//#include "SEMdata.hpp"
#include "SEMQkGLBasisFunctions.hpp"
#include "common/CArray.hpp"

#include<stdio.h>

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
class SEMQkGLIntegralsClassic
{
public:
  static constexpr int order = ORDER;
  static constexpr int ROW = (order + 1) * (order + 1) * (order + 1);


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
          int const q = q1 + q2 * (order + 1) + q3 * (order + 1) * (order + 1);

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
            int const a = a1 + q2*2 + q3*4;
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
            int const a = q1 + a2*2 + q3*4;
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
            int const a = q1 + q2*2 + a3*4;
            double X = coords( a, 0 );
            double Y = coords( a, 1 );
            double Z = coords( a, 2 );
            jac02 += X * dPhi[q3][a3];
            jac12 += Y * dPhi[q3][a3];
            jac22 += Z * dPhi[q3][a3];
          }

          // detJ
          double detJ = jac00 * (jac11 * jac22 - jac21 * jac12)
                             - jac01 * (jac10 * jac22 - jac20 * jac12)
                             + jac02 * (jac10 * jac21 - jac20 * jac11);

          // inv of jac is equal of the minors of the transposed of jac
          double invJac00 = jac11 * jac22 - jac12 * jac21;
          double invJac01 = jac02 * jac21 - jac01 * jac22;
          double invJac02 = jac01 * jac12 - jac02 * jac11;
          double invJac10 = jac12 * jac20 - jac10 * jac22;
          double invJac11 = jac00 * jac22 - jac02 * jac20;
          double invJac12 = jac02 * jac10 - jac00 * jac12;
          double invJac20 = jac10 * jac21 - jac11 * jac20;
          double invJac21 = jac01 * jac20 - jac00 * jac21;
          double invJac22 = jac00 * jac11 - jac01 * jac10;

          double transpInvJac00 = invJac00;
          double transpInvJac01 = invJac10;
          double transpInvJac02 = invJac20;
          double transpInvJac10 = invJac01;
          double transpInvJac11 = invJac11;
          double transpInvJac12 = invJac21;
          double transpInvJac20 = invJac02;
          double transpInvJac21 = invJac12;
          double transpInvJac22 = invJac22;

          double detJM1 = 1. / detJ;

          // B
          B[q][0] = (invJac00 * transpInvJac00 + invJac01 * transpInvJac10 + invJac02 * transpInvJac20) * detJM1; //B11
          B[q][1] = (invJac10 * transpInvJac01 + invJac11 * transpInvJac11 + invJac12 * transpInvJac21) * detJM1; //B22
          B[q][2] = (invJac20 * transpInvJac02 + invJac21 * transpInvJac12 + invJac22 * transpInvJac22) * detJM1; //B33
          B[q][3] = (invJac00 * transpInvJac01 + invJac01 * transpInvJac11 + invJac02 * transpInvJac21) * detJM1; //B12,B21
          B[q][4] = (invJac00 * transpInvJac02 + invJac01 * transpInvJac12 + invJac02 * transpInvJac22) * detJM1; //B13,B31
          B[q][5] = (invJac10 * transpInvJac02 + invJac11 * transpInvJac12 + invJac12 * transpInvJac22) * detJM1; //B23,B32

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
    float R[ROW];

    int orderPow2 = (ORDER + 1) * (ORDER + 1);
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
            int j = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
            for ( int l = 0; l < ORDER + 1; l++ )
            {
              int ll = l + i2 * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[l]* weights[i2] * weights[i3] * (B[ll][0] * dPhi[i1][l ]* dPhi[j1][l ]);
            }
          }
          //B22
          for ( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            int j = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
            for ( int m = 0; m < ORDER + 1; m++ )
            {
              int mm = i1 + m * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[i1] * weights[m]* weights[i3] * (B[mm][1] * dPhi[i2][m ]* dPhi[j2][m ]);
            }
          }
          //B33
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            int j = i1 + i2 * (ORDER + 1) + j3 * orderPow2;
            for ( int n = 0; n < ORDER + 1; n++ )
            {
              int nn = i1 + i2 * (ORDER + 1) + n * orderPow2;
              R[j] += weights[i1] * weights[i2] * weights[n]* (B[nn][2] * dPhi[i3][n ]* dPhi[j3][n ]);
            }
          }
          // B12,B21 (B[][3])
          for ( int j2 = 0; j2 < ORDER + 1; j2++ )
          {
            for ( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = j1 + j2 * (ORDER + 1) + i3 * orderPow2;
              int k = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int l = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
              R[j] += weights[j1] * weights[i2] * weights[i3] * (B[k][3] * dPhi[i1][j1] * dPhi[j2][i2] ) +
                      weights[i1] * weights[j2] * weights[i3] * (B[l][3] * dPhi[j1][i1] * dPhi[i2][j2] );
            }
          }
          // B13,B31 (B[][4])
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for ( int j1 = 0; j1 < ORDER + 1; j1++ )
            {
              int j = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int k = j1 + i2 * (ORDER + 1) + i3 * orderPow2;
              int l = j1 + i2 * (ORDER + 1) + j3 * orderPow2;
              R[j] += weights[j1] * weights[i2] * weights[i3] * (B[k][4] * dPhi[j1][i1] * dPhi[j3][i3] ) +
                      weights[j1] * weights[i2] * weights[j3] * (B[l][4] * dPhi[j1][i1] * dPhi[i3][j3] );
            }
          }
          // B23,B32 (B[][5])
          for ( int j3 = 0; j3 < ORDER + 1; j3++ )
          {
            for ( int j2 = 0; j2 < ORDER + 1; j2++ )
            {
              int j = i1 + j2 * (ORDER + 1) + j3 * orderPow2;
              int k = i1 + j2 * (ORDER + 1) + i3 * orderPow2;
              int l = i1 + i2 * (ORDER + 1) + j3 * orderPow2;
              R[j] += weights[i1] * weights[j2] * weights[i3] * (B[k][5] * dPhi[i2][i2] * dPhi[j3][i3] ) +
                      weights[i1] * weights[i2] * weights[j3] * (B[l][5] * dPhi[j2][i2] * dPhi[i3][j3] );
            }
          }

          int i = i1 + i2 * (ORDER + 1) + i3 * orderPow2;
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
  constexpr static void computeMassMatrixAndStiffnessVector( const int & elementNumber,
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
            int l = i + j * 2 + k * (2) * (2);
            X[I][0] = nodesCoordsX( elementNumber, l );
            X[I][1] = nodesCoordsY( elementNumber, l );
            X[I][2] = nodesCoordsZ( elementNumber, l );
            ++I;
          }
        }
      }
    }


    // ***** Compute Low order Jacobian and B-matrix *****
    double parentCoords[ORDER+1];
    double weights[ORDER+1];
    double dPhiO1[ORDER+1][2];

    SEMQkGLBasisFunctions<ORDER>::gaussLobattoQuadraturePoints( parentCoords );
    SEMQkGLBasisFunctions<ORDER>::getDerivativeBasisFunction1DLow( parentCoords, dPhiO1 );
    SEMQkGLBasisFunctions<ORDER>::gaussLobattoQuadratureWeights( weights );

    float B[ROW][6];
    computeB( weights, X, dPhiO1, massMatrixLocal, B );

    // ***** Compute High order gradient operations for Stiffness *****
    double dPhi[ORDER+1][ORDER+1];
    SEMQkGLBasisFunctions<ORDER>::getDerivativeBasisFunction1D( parentCoords, dPhi );

    // compute stifness  matrix ( durufle's optimization)
    gradPhiGradPhi( nPointsPerElement, weights, dPhi, B, pnLocal, Y );
  }

  /////////////////////////////////////////////////////////////////////////////////////
  //  end from first implementation
  /////////////////////////////////////////////////////////////////////////////////////

};

#endif //SEMQKGLINTEGRALSCLASSIC_HPP_
