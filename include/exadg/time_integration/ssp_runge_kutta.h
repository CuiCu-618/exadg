/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_SSP_RUNGE_KUTTA_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_SSP_RUNGE_KUTTA_H_

// deal.II
#include <deal.II/lac/full_matrix.h>

// ExaDG
#include <exadg/time_integration/explicit_runge_kutta.h>

namespace ExaDG
{
/*
 *  Strong-Stability-Preserving Runge-Kutta Methods according to
 *
 *    Kubatko et al., Optimal Strong-Stability-Preserving Runge-Kutta Time Discretizations
 *    for Discontinuous Galerkin Methods, J Sci Comput (2014) 60:313-344.
 *
 *    The Runge-Kutta scheme is implemented in Shu-Osher form instead of the Butcher form.
 *
 */
template<typename Operator, typename VectorType>
class SSPRK : public ExplicitTimeIntegrator<Operator, VectorType>
{
public:
  SSPRK(std::shared_ptr<Operator> const operator_in,
        unsigned int const              order_in,
        unsigned int const              stages_in)
    : ExplicitTimeIntegrator<Operator, VectorType>(operator_in), order(order_in)
  {
    initialize_coeffs(stages_in);
  }

  void
  solve_timestep(VectorType & vec_np,
                 VectorType & vec_n,
                 double const time,
                 double const time_step) final;

  unsigned int
  get_order() const final
  {
    return order;
  }

private:
  void
  initialize_coeffs(unsigned int const stages);

  dealii::FullMatrix<double> A, B;
  std::vector<double>        c;
  unsigned int const         order;

  std::vector<VectorType> u_vec, F_vec;
};

template<typename Operator, typename VectorType>
void
SSPRK<Operator, VectorType>::solve_timestep(VectorType & vec_np,
                                            VectorType & vec_n,
                                            double const time,
                                            double const time_step)
{
  unsigned int const stages = A.m();

  // Initialize vectors if necessary
  if(u_vec.empty() or not u_vec[0].partitioners_are_globally_compatible(*vec_n.get_partitioner()))
  {
    u_vec.resize(stages + 1);
    for(unsigned int d = 0; d < u_vec.size(); ++d)
      u_vec[d].reinit(vec_n);

    F_vec.resize(stages);
    for(unsigned int d = 0; d < F_vec.size(); ++d)
      F_vec[d].reinit(vec_n, true);
  }

  u_vec[0] = vec_n;

  for(unsigned int s = 1; s <= stages; ++s)
  {
    this->underlying_operator->evaluate(F_vec[s - 1], u_vec[s - 1], time + c[s - 1] * time_step);

    u_vec[s] = 0.; // Do not forget to reset u_vec[s]!
    for(unsigned int l = 0; l < s; ++l)
    {
      u_vec[s].add(A[s - 1][l], u_vec[l], B[s - 1][l] * time_step, F_vec[l]);
    }
  }

  vec_np = u_vec[stages];
}

template<typename Operator, typename VectorType>
void
SSPRK<Operator, VectorType>::initialize_coeffs(unsigned int const stages)
{
  A.reinit(stages, stages);
  B.reinit(stages, stages);

  bool coefficients_are_initialized = false;

  if(order == 3)
  {
    if(stages == 4)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[1][0] = 0.522361915162541;
      A[1][1] = 0.477638084837459;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[2][0] = 0.368530939472566;
      A[2][1] = 0.;
      A[2][2] = 0.631469060527434;
      A[2][3] = 0.;
      A[3][0] = 0.334082932462285;
      A[3][1] = 0.006966183666289;
      A[3][2] = 0.;
      A[3][3] = 0.658950883871426;

      B[0][0] = 0.594057152884440;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.283744320787718;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[2][0] = 0.000000038023030;
      B[2][1] = 0.;
      B[2][2] = 0.375128712231540;
      B[2][3] = 0.;
      B[3][0] = 0.116941419604231;
      B[3][1] = 0.004138311235266;
      B[3][2] = 0.;
      B[3][3] = 0.391454485963345;

      coefficients_are_initialized = true;
    }
    else if(stages == 5)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[1][0] = 0.495124140877703;
      A[1][1] = 0.504875859122297;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[2][0] = 0.105701991897526;
      A[2][1] = 0.;
      A[2][2] = 0.894298008102474;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[3][0] = 0.411551205755676;
      A[3][1] = 0.011170516177380;
      A[3][2] = 0.;
      A[3][3] = 0.577278278066944;
      A[3][4] = 0.;
      A[4][0] = 0.186911123548222;
      A[4][1] = 0.013354480555382;
      A[4][2] = 0.012758264566319;
      A[4][3] = 0.;
      A[4][4] = 0.786976131330077;

      B[0][0] = 0.418883109982196;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.211483970024081;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[2][0] = 0.000000000612488;
      B[2][1] = 0.;
      B[2][2] = 0.374606330884848;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[3][0] = 0.046744815663888;
      B[3][1] = 0.004679140556487;
      B[3][2] = 0.;
      B[3][3] = 0.241812120441849;
      B[3][4] = 0.;
      B[4][0] = 0.071938257223857;
      B[4][1] = 0.005593966347235;
      B[4][2] = 0.005344221539515;
      B[4][3] = 0.;
      B[4][4] = 0.329651009373300;

      coefficients_are_initialized = true;
    }
    else if(stages == 6)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[1][0] = 0.271376652410776;
      A[1][1] = 0.728623347589224;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[2][0] = 0.003607665467954;
      A[2][1] = 0.;
      A[2][2] = 0.996392334532046;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[3][0] = 0.295174024904477;
      A[3][1] = 0.104490494022953;
      A[3][2] = 0.;
      A[3][3] = 0.600335481072570;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[4][0] = 0.300088895805571;
      A[4][1] = 0.000000004174982;
      A[4][2] = 0.000038417983374;
      A[4][3] = 0.;
      A[4][4] = 0.699872682036073;
      A[4][5] = 0.;
      A[5][0] = 0.057902281374384;
      A[5][1] = 0.003951957060919;
      A[5][2] = 0.179481122980769;
      A[5][3] = 0.126656280556504;
      A[5][4] = 0.;
      A[5][5] = 0.632008358027424;

      B[0][0] = 0.325620674236780;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.237254825706663;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[2][0] = 0.000014278868889;
      B[2][1] = 0.;
      B[2][2] = 0.324445943775684;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[3][0] = 0.000000008816565;
      B[3][1] = 0.034024265115088;
      B[3][2] = 0.;
      B[3][3] = 0.195481644115112;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[4][0] = 0.0;
      B[4][1] = 0.000000001359460;
      B[4][2] = 0.000012509689649;
      B[4][3] = 0.;
      B[4][4] = 0.227893014604489;
      B[4][5] = 0.;
      B[5][0] = 0.033480821651945;
      B[5][1] = 0.001286838922731;
      B[5][2] = 0.058442764277772;
      B[5][3] = 0.041241903471131;
      B[5][4] = 0.;
      B[5][5] = 0.205794987664170;

      coefficients_are_initialized = true;
    }
    else if(stages == 7)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[0][6] = 0.;
      A[1][0] = 0.412429019730110;
      A[1][1] = 0.587570980269890;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[1][6] = 0.;
      A[2][0] = 0.005800594241485;
      A[2][1] = 0.0;
      A[2][2] = 0.994199405758515;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[2][6] = 0.;
      A[3][0] = 0.162485678538202;
      A[3][1] = 0.000000000270334;
      A[3][2] = 0.;
      A[3][3] = 0.837514321191464;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[3][6] = 0.;
      A[4][0] = 0.205239611567914;
      A[4][1] = 0.000000000554433;
      A[4][2] = 0.001461982584386;
      A[4][3] = 0.;
      A[4][4] = 0.793298405293266;
      A[4][5] = 0.;
      A[4][6] = 0.;
      A[5][0] = 0.246951813330533;
      A[5][1] = 0.000686077138452;
      A[5][2] = 0.098274672761128;
      A[5][3] = 0.125080337194733;
      A[5][4] = 0.;
      A[5][5] = 0.529007099575153;
      A[5][6] = 0.;
      A[6][0] = 0.003515397992512;
      A[6][1] = 0.002051029751004;
      A[6][2] = 0.037621575915744;
      A[6][3] = 0.113733937331291;
      A[6][4] = 0.000552268540167;
      A[6][5] = 0.;
      A[6][6] = 0.842525790469282;

      B[0][0] = 0.267322588523961;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[0][6] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.157070995387308;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[1][6] = 0.;
      B[2][0] = 0.019051847781300;
      B[2][1] = 0.;
      B[2][2] = 0.265771958656350;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[2][6] = 0.;
      B[3][0] = 0.014327744686556;
      B[3][1] = 0.000000000072266;
      B[3][2] = 0.;
      B[3][3] = 0.223886496266790;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[3][6] = 0.;
      B[4][0] = 0.030979976588062;
      B[4][1] = 0.000000000148213;
      B[4][2] = 0.000390820968835;
      B[4][3] = 0.;
      B[4][4] = 0.212066583174926;
      B[4][5] = 0.;
      B[4][6] = 0.;
      B[5][0] = 0.004054481853252;
      B[5][1] = 0.000183403916578;
      B[5][2] = 0.026271039908850;
      B[5][3] = 0.033436799512346;
      B[5][4] = 0.;
      B[5][5] = 0.141415547205983;
      B[5][6] = 0.;
      B[6][0] = 0.021050441338920;
      B[6][1] = 0.000548286582178;
      B[6][2] = 0.010057097058147;
      B[6][3] = 0.030403650530423;
      B[6][4] = 0.000147633855718;
      B[6][5] = 0.;
      B[6][6] = 0.225226175206445;

      coefficients_are_initialized = true;
    }
    else if(stages == 8)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[0][6] = 0.;
      A[0][7] = 0.;
      A[1][0] = 0.108675201424538;
      A[1][1] = 0.891324798575462;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[1][6] = 0.;
      A[1][7] = 0.;
      A[2][0] = 0.008159777689219;
      A[2][1] = 0.;
      A[2][2] = 0.991840222310781;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[2][6] = 0.;
      A[2][7] = 0.;
      A[3][0] = 0.000075204616622;
      A[3][1] = 0.000017473454611;
      A[3][2] = 0.;
      A[3][3] = 0.999907321928768;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[3][6] = 0.;
      A[3][7] = 0.;
      A[4][0] = 0.275083494553101;
      A[4][1] = 0.013251614514063;
      A[4][2] = 0.333930523474093;
      A[4][3] = 0.;
      A[4][4] = 0.377734367458743;
      A[4][5] = 0.;
      A[4][6] = 0.;
      A[4][7] = 0.;
      A[5][0] = 0.172210423641858;
      A[5][1] = 0.067723791902171;
      A[5][2] = 0.031061316699451;
      A[5][3] = 0.018868041432255;
      A[5][4] = 0.;
      A[5][5] = 0.710136426324266;
      A[5][6] = 0.;
      A[5][7] = 0.;
      A[6][0] = 0.155954681117895;
      A[6][1] = 0.;
      A[6][2] = 0.000000000164948;
      A[6][3] = 0.000000000009768;
      A[6][4] = 0.000000000001983;
      A[6][5] = 0.;
      A[6][6] = 0.844045318705406;
      A[6][7] = 0.;
      A[7][0] = 0.021413729448041;
      A[7][1] = 0.000000000008708;
      A[7][2] = 0.000000000028479;
      A[7][3] = 0.072111559681400;
      A[7][4] = 0.109489249417096;
      A[7][5] = 0.046882143587611;
      A[7][6] = 0.;
      A[7][7] = 0.750103317828665;

      B[0][0] = 0.227519284497891;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[0][6] = 0.;
      B[0][7] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.202793580427116;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[1][6] = 0.;
      B[1][7] = 0.;
      B[2][0] = 0.026257801089696;
      B[2][1] = 0.;
      B[2][2] = 0.225662777716378;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[2][6] = 0.;
      B[2][7] = 0.;
      B[3][0] = 0.000000000064040;
      B[3][1] = 0.000003975547891;
      B[3][2] = 0.;
      B[3][3] = 0.227498198449436;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[3][6] = 0.;
      B[3][7] = 0.;
      B[4][0] = 0.006914605853513;
      B[4][1] = 0.003014997852682;
      B[4][2] = 0.075975633772832;
      B[4][3] = 0.;
      B[4][4] = 0.085941853014477;
      B[4][5] = 0.;
      B[4][6] = 0.;
      B[4][7] = 0.;
      B[5][0] = 0.000733937709508;
      B[5][1] = 0.015408468677066;
      B[5][2] = 0.007067048551022;
      B[5][3] = 0.004292843286543;
      B[5][4] = 0.;
      B[5][5] = 0.161569731613186;
      B[5][6] = 0.;
      B[5][7] = 0.;
      B[6][0] = 0.000000000000089;
      B[6][1] = 0.;
      B[6][2] = 0.000000000037529;
      B[6][3] = 0.000000000002222;
      B[6][4] = 0.000000000000451;
      B[6][5] = 0.;
      B[6][6] = 0.192036586995649;
      B[6][7] = 0.;
      B[7][0] = 0.024945755721405;
      B[7][1] = 0.000000000001981;
      B[7][2] = 0.000000000006480;
      B[7][3] = 0.016406770462739;
      B[7][4] = 0.024910915687589;
      B[7][5] = 0.010666591764781;
      B[7][6] = 0.;
      B[7][7] = 0.170662970171872;

      coefficients_are_initialized = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }
  else if(order == 4)
  {
    if(stages == 5)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[1][0] = 0.261216512493821;
      A[1][1] = 0.738783487506179;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[2][0] = 0.623613752757655;
      A[2][1] = 0.;
      A[2][2] = 0.376386247242345;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[3][0] = 0.444745181201454;
      A[3][1] = 0.120932584902288;
      A[3][2] = 0.;
      A[3][3] = 0.434322233896258;
      A[3][4] = 0.;
      A[4][0] = 0.213357715199957;
      A[4][1] = 0.209928473023448;
      A[4][2] = 0.063353148180384;
      A[4][3] = 0.;
      A[4][4] = 0.513360663596212;

      B[0][0] = 0.605491839566400;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.447327372891397;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[2][0] = 0.000000844149769;
      B[2][1] = 0.;
      B[2][2] = 0.227898801230261;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[3][0] = 0.002856233144485;
      B[3][1] = 0.073223693296006;
      B[3][2] = 0.;
      B[3][3] = 0.262978568366434;
      B[3][4] = 0.;
      B[4][0] = 0.002362549760441;
      B[4][1] = 0.127109977308333;
      B[4][2] = 0.038359814234063;
      B[4][3] = 0.;
      B[4][4] = 0.310835692561898;

      coefficients_are_initialized = true;
    }
    else if(stages == 6)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[1][0] = 0.441581886978406;
      A[1][1] = 0.558418113021594;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[2][0] = 0.496140382330059;
      A[2][1] = 0.;
      A[2][2] = 0.503859617669941;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[3][0] = 0.392013998230666;
      A[3][1] = 0.001687525300458;
      A[3][2] = -0.;
      A[3][3] = 0.606298476468875;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[4][0] = 0.016884674246355;
      A[4][1] = 0.000000050328214;
      A[4][2] = 0.000018549175549;
      A[4][3] = 0.;
      A[4][4] = 0.983096726249882;
      A[4][5] = 0.;
      A[5][0] = 0.128599802059752;
      A[5][1] = 0.150433518466544;
      A[5][2] = 0.179199506866483;
      A[5][3] = 0.173584325551242;
      A[5][4] = 0.;
      A[5][5] = 0.368182847055979;

      B[0][0] = 0.448860018455995;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.250651564517035;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[2][0] = 0.004050697317371;
      B[2][1] = 0.;
      B[2][2] = 0.226162437286560;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[3][0] = 0.000000073512372;
      B[3][1] = 0.000757462637509;
      B[3][2] = -0.;
      B[3][3] = 0.272143145337661;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[4][0] = 0.000592927398846;
      B[4][1] = 0.000000022590323;
      B[4][2] = 0.000008325983279;
      B[4][3] = 0.;
      B[4][4] = 0.441272814688551;
      B[4][5] = 0.;
      B[5][0] = 0.000000009191468;
      B[5][1] = 0.067523591875293;
      B[5][2] = 0.080435493959395;
      B[5][3] = 0.077915063570602;
      B[5][4] = 0.;
      B[5][5] = 0.165262559524728;

      coefficients_are_initialized = true;
    }
    else if(stages == 7)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[0][6] = 0.;
      A[1][0] = 0.277584603405600;
      A[1][1] = 0.722415396594400;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[1][6] = 0.;
      A[2][0] = 0.528403304637363;
      A[2][1] = 0.018109310473034;
      A[2][2] = 0.453487384889603;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[2][6] = 0.;
      A[3][0] = 0.363822566916605;
      A[3][1] = 0.025636760093079;
      A[3][2] = 0.000072932527637;
      A[3][3] = 0.610467740462679;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[3][6] = 0.;
      A[4][0] = 0.080433061177282;
      A[4][1] = 0.000000001538366;
      A[4][2] = 0.000000000000020;
      A[4][3] = 0.000000000036824;
      A[4][4] = 0.919566937247508;
      A[4][5] = 0.;
      A[4][6] = 0.;
      A[5][0] = 0.305416318145737;
      A[5][1] = 0.017282647045059;
      A[5][2] = 0.214348299745317;
      A[5][3] = 0.001174022148498;
      A[5][4] = 0.003799138070873;
      A[5][5] = 0.457979574844515;
      A[5][6] = 0.;
      A[6][0] = 0.112741543203136;
      A[6][1] = 0.042888410429255;
      A[6][2] = 0.185108001868376;
      A[6][3] = 0.000003952121250;
      A[6][4] = 0.230275526732661;
      A[6][5] = 0.110240916986851;
      A[6][6] = 0.318741648658470;

      B[0][0] = 0.236998129331275;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[0][6] = 0.;
      B[1][0] = 0.001205136607466;
      B[1][1] = 0.310012922173259;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[1][6] = 0.;
      B[2][0] = 0.000000000029361;
      B[2][1] = 0.007771318668946;
      B[2][2] = 0.194606801046999;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[2][6] = 0.;
      B[3][0] = 0.001612059039346;
      B[3][1] = 0.011001602331536;
      B[3][2] = 0.000031297818569;
      B[3][3] = 0.261972390131100;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[3][6] = 0.;
      B[4][0] = 0.000000000027723;
      B[4][1] = 0.000000000660165;
      B[4][2] = 0.000000000000009;
      B[4][3] = 0.000000000015802;
      B[4][4] = 0.394617327778342;
      B[4][5] = 0.;
      B[4][6] = 0.;
      B[5][0] = 0.115125889382648;
      B[5][1] = 0.007416569384575;
      B[5][2] = 0.091984117559200;
      B[5][3] = 0.000503812679890;
      B[5][4] = 0.001630338861330;
      B[5][5] = 0.196534551952426;
      B[5][6] = 0.;
      B[6][0] = 0.000102167855778;
      B[6][1] = 0.018404869978158;
      B[6][2] = 0.079436115076445;
      B[6][3] = 0.000001695989127;
      B[6][4] = 0.098819030275264;
      B[6][5] = 0.047308112450629;
      B[6][6] = 0.136782840433305;

      coefficients_are_initialized = true;
    }
    else if(stages == 8)
    {
      A[0][0] = 1.;
      A[0][1] = 0.;
      A[0][2] = 0.;
      A[0][3] = 0.;
      A[0][4] = 0.;
      A[0][5] = 0.;
      A[0][6] = 0.;
      A[0][7] = 0.;
      A[1][0] = 0.538569155333175;
      A[1][1] = 0.461430844666825;
      A[1][2] = 0.;
      A[1][3] = 0.;
      A[1][4] = 0.;
      A[1][5] = 0.;
      A[1][6] = 0.;
      A[1][7] = 0.;
      A[2][0] = 0.004485387460763;
      A[2][1] = 0.;
      A[2][2] = 0.995514612539237;
      A[2][3] = 0.;
      A[2][4] = 0.;
      A[2][5] = 0.;
      A[2][6] = 0.;
      A[2][7] = 0.;
      A[3][0] = 0.164495299288580;
      A[3][1] = 0.016875060685979;
      A[3][2] = 0.;
      A[3][3] = 0.818629640025440;
      A[3][4] = 0.;
      A[3][5] = 0.;
      A[3][6] = 0.;
      A[3][7] = 0.;
      A[4][0] = 0.426933682982668;
      A[4][1] = 0.157047028197878;
      A[4][2] = 0.023164224070770;
      A[4][3] = 0.;
      A[4][4] = 0.392855064748685;
      A[4][5] = 0.;
      A[4][6] = 0.;
      A[4][7] = 0.;
      A[5][0] = 0.082083400476958;
      A[5][1] = 0.000000039091042;
      A[5][2] = 0.033974171137350;
      A[5][3] = 0.005505195713107;
      A[5][4] = 0.;
      A[5][5] = 0.878437193581543;
      A[5][6] = 0.;
      A[5][7] = 0.;
      A[6][0] = 0.006736365648625;
      A[6][1] = 0.010581829625529;
      A[6][2] = 0.009353386191951;
      A[6][3] = 0.101886062556838;
      A[6][4] = 0.000023428364930;
      A[6][5] = 0.;
      A[6][6] = 0.871418927612128;
      A[6][7] = 0.;
      A[7][0] = 0.071115287415749;
      A[7][1] = 0.018677648343953;
      A[7][2] = 0.007902408660034;
      A[7][3] = 0.319384027162348;
      A[7][4] = 0.007121989995845;
      A[7][5] = 0.001631615692736;
      A[7][6] = 0.;
      A[7][7] = 0.574167022729334;

      B[0][0] = 0.282318339066479;
      B[0][1] = 0.;
      B[0][2] = 0.;
      B[0][3] = 0.;
      B[0][4] = 0.;
      B[0][5] = 0.;
      B[0][6] = 0.;
      B[0][7] = 0.;
      B[1][0] = 0.;
      B[1][1] = 0.130270389660380;
      B[1][2] = 0.;
      B[1][3] = 0.;
      B[1][4] = 0.;
      B[1][5] = 0.;
      B[1][6] = 0.;
      B[1][7] = 0.;
      B[2][0] = 0.003963092203460;
      B[2][1] = 0.;
      B[2][2] = 0.281052031928487;
      B[2][3] = 0.;
      B[2][4] = 0.;
      B[2][5] = 0.;
      B[2][6] = 0.;
      B[2][7] = 0.;
      B[3][0] = 0.000038019518678;
      B[3][1] = 0.004764139104512;
      B[3][2] = 0.;
      B[3][3] = 0.231114160282572;
      B[3][4] = 0.;
      B[3][5] = 0.;
      B[3][6] = 0.;
      B[3][7] = 0.;
      B[4][0] = 0.000019921336144;
      B[4][1] = 0.044337256156151;
      B[4][2] = 0.006539685265423;
      B[4][3] = 0.;
      B[4][4] = 0.110910189373703;
      B[4][5] = 0.;
      B[4][6] = 0.;
      B[4][7] = 0.;
      B[5][0] = 0.000000034006679;
      B[5][1] = 0.000000011036118;
      B[5][2] = 0.009591531566657;
      B[5][3] = 0.001554217709960;
      B[5][4] = 0.;
      B[5][5] = 0.247998929466160;
      B[5][6] = 0.;
      B[5][7] = 0.;
      B[6][0] = 0.013159891155054;
      B[6][1] = 0.002987444564164;
      B[6][2] = 0.002640632454359;
      B[6][3] = 0.028764303955070;
      B[6][4] = 0.000006614257074;
      B[6][5] = 0.;
      B[6][6] = 0.246017544274548;
      B[6][7] = 0.;
      B[7][0] = 0.000000010647874;
      B[7][1] = 0.005273042658132;
      B[7][2] = 0.002230994887525;
      B[7][3] = 0.090167968072837;
      B[7][4] = 0.002010668386475;
      B[7][5] = 0.000460635032368;
      B[7][6] = 0.;
      B[7][7] = 0.162097880203691;

      coefficients_are_initialized = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }

  AssertThrow(coefficients_are_initialized, dealii::ExcNotImplemented());

  // calculate coefficients c_i for non-autonomous systems, i.e.,
  // systems with explicit time-dependency
  dealii::FullMatrix<double> crk;
  crk.reinit(stages, stages);

  for(unsigned int k = 0; k < stages; ++k)
  {
    for(unsigned int i = 0; i < stages; ++i)
    {
      double csum = 0.;
      for(unsigned int l = k + 1; l < i + 1; ++l)
      {
        csum = csum + crk[l - 1][k] * A[i][l];
      }
      crk[i][k] = B[i][k] + csum;
    }
  }

  c.resize(stages);
  c[0] = 0.;
  for(unsigned int k = 1; k < stages; ++k)
  {
    c[k] = 0.;
    for(unsigned int l = 0; l < k + 1; ++l)
    {
      c[k] = c[k] + crk[k - 1][l];
    }
  }
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_SSP_RUNGE_KUTTA_H_ */
