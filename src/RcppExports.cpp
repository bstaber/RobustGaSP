// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/RobustGaSP.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// matern_5_2_funct
Eigen::MatrixXd matern_5_2_funct(const MapMat& d, double beta_i);
RcppExport SEXP _RobustGaSP_matern_5_2_funct(SEXP dSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_5_2_funct(d, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// matern_3_2_funct
Eigen::MatrixXd matern_3_2_funct(const Eigen::Map<Eigen::MatrixXd>& d, double beta_i);
RcppExport SEXP _RobustGaSP_matern_3_2_funct(SEXP dSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_3_2_funct(d, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// pow_exp_funct
Eigen::MatrixXd pow_exp_funct(const MapMat& d, double beta_i, double alpha_i);
RcppExport SEXP _RobustGaSP_pow_exp_funct(SEXP dSEXP, SEXP beta_iSEXP, SEXP alpha_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    Rcpp::traits::input_parameter< double >::type alpha_i(alpha_iSEXP);
    rcpp_result_gen = Rcpp::wrap(pow_exp_funct(d, beta_i, alpha_i));
    return rcpp_result_gen;
END_RCPP
}
// periodic_gauss_funct
Eigen::MatrixXd periodic_gauss_funct(const MapMat& d, double beta_i);
RcppExport SEXP _RobustGaSP_periodic_gauss_funct(SEXP dSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(periodic_gauss_funct(d, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// periodic_exp_funct
Eigen::MatrixXd periodic_exp_funct(const MapMat& d, double beta_i);
RcppExport SEXP _RobustGaSP_periodic_exp_funct(SEXP dSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(periodic_exp_funct(d, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// matern_5_2_deriv
Eigen::MatrixXd matern_5_2_deriv(const MapMat& R0_i, const Mat R, double beta_i);
RcppExport SEXP _RobustGaSP_matern_5_2_deriv(SEXP R0_iSEXP, SEXP RSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type R0_i(R0_iSEXP);
    Rcpp::traits::input_parameter< const Mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_5_2_deriv(R0_i, R, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// matern_3_2_deriv
Eigen::MatrixXd matern_3_2_deriv(const Eigen::Map<Eigen::MatrixXd>& R0_i, const Eigen::MatrixXd R, double beta_i);
RcppExport SEXP _RobustGaSP_matern_3_2_deriv(SEXP R0_iSEXP, SEXP RSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R0_i(R0_iSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type R(RSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(matern_3_2_deriv(R0_i, R, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// pow_exp_deriv
Eigen::MatrixXd pow_exp_deriv(const MapMat& R0_i, const Eigen::MatrixXd R, const double beta_i, const double alpha_i);
RcppExport SEXP _RobustGaSP_pow_exp_deriv(SEXP R0_iSEXP, SEXP RSEXP, SEXP beta_iSEXP, SEXP alpha_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type R0_i(R0_iSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type R(RSEXP);
    Rcpp::traits::input_parameter< const double >::type beta_i(beta_iSEXP);
    Rcpp::traits::input_parameter< const double >::type alpha_i(alpha_iSEXP);
    rcpp_result_gen = Rcpp::wrap(pow_exp_deriv(R0_i, R, beta_i, alpha_i));
    return rcpp_result_gen;
END_RCPP
}
// periodic_gauss_deriv
Eigen::MatrixXd periodic_gauss_deriv(const MapMat& R0_i, const Eigen::MatrixXd& R, double beta_i);
RcppExport SEXP _RobustGaSP_periodic_gauss_deriv(SEXP R0_iSEXP, SEXP RSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type R0_i(R0_iSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type R(RSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(periodic_gauss_deriv(R0_i, R, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// periodic_exp_deriv
Eigen::MatrixXd periodic_exp_deriv(const MapMat& R0_i, const Eigen::MatrixXd& R, double beta_i);
RcppExport SEXP _RobustGaSP_periodic_exp_deriv(SEXP R0_iSEXP, SEXP RSEXP, SEXP beta_iSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type R0_i(R0_iSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type R(RSEXP);
    Rcpp::traits::input_parameter< double >::type beta_i(beta_iSEXP);
    rcpp_result_gen = Rcpp::wrap(periodic_exp_deriv(R0_i, R, beta_i));
    return rcpp_result_gen;
END_RCPP
}
// separable_kernel
Eigen::MatrixXd separable_kernel(List R0, Eigen::VectorXd beta, String kernel_type, Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_separable_kernel(SEXP R0SEXP, SEXP betaSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(separable_kernel(R0, beta, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// separable_multi_kernel
Eigen::MatrixXd separable_multi_kernel(List R0, Eigen::VectorXd beta, Eigen::VectorXi kernel_type, Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_separable_multi_kernel(SEXP R0SEXP, SEXP betaSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(separable_multi_kernel(R0, beta, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// separable_multi_kernel_pred_periodic
Eigen::MatrixXd separable_multi_kernel_pred_periodic(const List R0, const Eigen::VectorXd beta, const Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha, const Eigen::VectorXd perid_const);
RcppExport SEXP _RobustGaSP_separable_multi_kernel_pred_periodic(SEXP R0SEXP, SEXP betaSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP, SEXP perid_constSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type perid_const(perid_constSEXP);
    rcpp_result_gen = Rcpp::wrap(separable_multi_kernel_pred_periodic(R0, beta, kernel_type, alpha, perid_const));
    return rcpp_result_gen;
END_RCPP
}
// euclidean_distance
Eigen::MatrixXd euclidean_distance(const MapMat& input1, const MapMat& input2);
RcppExport SEXP _RobustGaSP_euclidean_distance(SEXP input1SEXP, SEXP input2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type input1(input1SEXP);
    Rcpp::traits::input_parameter< const MapMat& >::type input2(input2SEXP);
    rcpp_result_gen = Rcpp::wrap(euclidean_distance(input1, input2));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_lik
double log_marginal_lik(const Vec param, double nugget, const bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_marginal_lik(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< const bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_profile_lik
double log_profile_lik(const Vec param, double nugget, const bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_profile_lik(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< const bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_profile_lik(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_approx_ref_prior
double log_approx_ref_prior(const Vec param, double nugget, bool nugget_est, const Eigen::VectorXd CL, const double a, const double b);
RcppExport SEXP _RobustGaSP_log_approx_ref_prior(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP CLSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type CL(CLSEXP);
    Rcpp::traits::input_parameter< const double >::type a(aSEXP);
    Rcpp::traits::input_parameter< const double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(log_approx_ref_prior(param, nugget, nugget_est, CL, a, b));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_lik_deriv
Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_marginal_lik_deriv(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik_deriv(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_profile_lik_deriv
Eigen::VectorXd log_profile_lik_deriv(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_profile_lik_deriv(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_profile_lik_deriv(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_approx_ref_prior_deriv
Eigen::VectorXd log_approx_ref_prior_deriv(const Vec param, double nugget, bool nugget_est, const Eigen::VectorXd CL, const double a, const double b);
RcppExport SEXP _RobustGaSP_log_approx_ref_prior_deriv(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP CLSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type CL(CLSEXP);
    Rcpp::traits::input_parameter< const double >::type a(aSEXP);
    Rcpp::traits::input_parameter< const double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(log_approx_ref_prior_deriv(param, nugget, nugget_est, CL, a, b));
    return rcpp_result_gen;
END_RCPP
}
// log_ref_marginal_post
double log_ref_marginal_post(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_ref_marginal_post(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_ref_marginal_post(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// construct_rgasp
List construct_rgasp(const Eigen::VectorXd beta, const double nu, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_construct_rgasp(SEXP betaSEXP, SEXP nuSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(construct_rgasp(beta, nu, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// pred_rgasp
List pred_rgasp(const Eigen::VectorXd beta, const double nu, const Eigen::Map<Eigen::MatrixXd>& input, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const Eigen::Map<Eigen::MatrixXd>& testing_input, const Eigen::Map<Eigen::MatrixXd>& X_testing, const Eigen::Map<Eigen::MatrixXd>& L, Eigen::Map<Eigen::MatrixXd>& LX, Eigen::Map<Eigen::VectorXd>& theta_hat, double sigma2_hat, double q_025, double q_975, List r0, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha, const String method, const bool interval_data);
RcppExport SEXP _RobustGaSP_pred_rgasp(SEXP betaSEXP, SEXP nuSEXP, SEXP inputSEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP testing_inputSEXP, SEXP X_testingSEXP, SEXP LSEXP, SEXP LXSEXP, SEXP theta_hatSEXP, SEXP sigma2_hatSEXP, SEXP q_025SEXP, SEXP q_975SEXP, SEXP r0SEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP, SEXP methodSEXP, SEXP interval_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type input(inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type testing_input(testing_inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X_testing(X_testingSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type L(LSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type LX(LXSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd>& >::type theta_hat(theta_hatSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_hat(sigma2_hatSEXP);
    Rcpp::traits::input_parameter< double >::type q_025(q_025SEXP);
    Rcpp::traits::input_parameter< double >::type q_975(q_975SEXP);
    Rcpp::traits::input_parameter< List >::type r0(r0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< const bool >::type interval_data(interval_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(pred_rgasp(beta, nu, input, X, zero_mean, output, testing_input, X_testing, L, LX, theta_hat, sigma2_hat, q_025, q_975, r0, kernel_type, alpha, method, interval_data));
    return rcpp_result_gen;
END_RCPP
}
// generate_predictive_mean_cov
List generate_predictive_mean_cov(const Eigen::VectorXd beta, const double nu, const Eigen::Map<Eigen::MatrixXd>& input, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const Eigen::Map<Eigen::MatrixXd>& testing_input, const Eigen::Map<Eigen::MatrixXd>& X_testing, const Eigen::Map<Eigen::MatrixXd>& L, Eigen::Map<Eigen::MatrixXd>& LX, Eigen::Map<Eigen::VectorXd>& theta_hat, double sigma2_hat, List rr0, List r0, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha, const String method, const bool sample_data);
RcppExport SEXP _RobustGaSP_generate_predictive_mean_cov(SEXP betaSEXP, SEXP nuSEXP, SEXP inputSEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP testing_inputSEXP, SEXP X_testingSEXP, SEXP LSEXP, SEXP LXSEXP, SEXP theta_hatSEXP, SEXP sigma2_hatSEXP, SEXP rr0SEXP, SEXP r0SEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP, SEXP methodSEXP, SEXP sample_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type input(inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type testing_input(testing_inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X_testing(X_testingSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type L(LSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type LX(LXSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd>& >::type theta_hat(theta_hatSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_hat(sigma2_hatSEXP);
    Rcpp::traits::input_parameter< List >::type rr0(rr0SEXP);
    Rcpp::traits::input_parameter< List >::type r0(r0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< const bool >::type sample_data(sample_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(generate_predictive_mean_cov(beta, nu, input, X, zero_mean, output, testing_input, X_testing, L, LX, theta_hat, sigma2_hat, rr0, r0, kernel_type, alpha, method, sample_data));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_lik_ppgasp
double log_marginal_lik_ppgasp(const Vec param, double nugget, const bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_marginal_lik_ppgasp(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< const bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik_ppgasp(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_profile_lik_ppgasp
double log_profile_lik_ppgasp(const Vec param, double nugget, const bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_profile_lik_ppgasp(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Vec >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< const bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_profile_lik_ppgasp(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_ref_marginal_post_ppgasp
double log_ref_marginal_post_ppgasp(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_ref_marginal_post_ppgasp(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_ref_marginal_post_ppgasp(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_lik_deriv_ppgasp
Eigen::VectorXd log_marginal_lik_deriv_ppgasp(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_marginal_lik_deriv_ppgasp(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik_deriv_ppgasp(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_profile_lik_deriv_ppgasp
Eigen::VectorXd log_profile_lik_deriv_ppgasp(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_log_profile_lik_deriv_ppgasp(SEXP paramSEXP, SEXP nuggetSEXP, SEXP nugget_estSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type param(paramSEXP);
    Rcpp::traits::input_parameter< double >::type nugget(nuggetSEXP);
    Rcpp::traits::input_parameter< bool >::type nugget_est(nugget_estSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_profile_lik_deriv_ppgasp(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// construct_ppgasp
List construct_ppgasp(const Eigen::VectorXd beta, const double nu, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_construct_ppgasp(SEXP betaSEXP, SEXP nuSEXP, SEXP R0SEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const List >::type R0(R0SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(construct_ppgasp(beta, nu, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// pred_ppgasp
List pred_ppgasp(const Eigen::VectorXd beta, const double nu, const Eigen::Map<Eigen::MatrixXd>& input, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const Eigen::Map<Eigen::MatrixXd>& testing_input, const Eigen::Map<Eigen::MatrixXd>& X_testing, const Eigen::Map<Eigen::MatrixXd>& L, Eigen::Map<Eigen::MatrixXd>& LX, Eigen::Map<Eigen::MatrixXd>& theta_hat, const Eigen::Map<Eigen::VectorXd>& sigma2_hat, double q_025, double q_975, List r0, Eigen::VectorXi kernel_type, const Eigen::VectorXd alpha, const String method, const bool interval_data);
RcppExport SEXP _RobustGaSP_pred_ppgasp(SEXP betaSEXP, SEXP nuSEXP, SEXP inputSEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP testing_inputSEXP, SEXP X_testingSEXP, SEXP LSEXP, SEXP LXSEXP, SEXP theta_hatSEXP, SEXP sigma2_hatSEXP, SEXP q_025SEXP, SEXP q_975SEXP, SEXP r0SEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP, SEXP methodSEXP, SEXP interval_dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type input(inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const String >::type zero_mean(zero_meanSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type output(outputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type testing_input(testing_inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X_testing(X_testingSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type L(LSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type LX(LXSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type theta_hat(theta_hatSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd>& >::type sigma2_hat(sigma2_hatSEXP);
    Rcpp::traits::input_parameter< double >::type q_025(q_025SEXP);
    Rcpp::traits::input_parameter< double >::type q_975(q_975SEXP);
    Rcpp::traits::input_parameter< List >::type r0(r0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< const bool >::type interval_data(interval_dataSEXP);
    rcpp_result_gen = Rcpp::wrap(pred_ppgasp(beta, nu, input, X, zero_mean, output, testing_input, X_testing, L, LX, theta_hat, sigma2_hat, q_025, q_975, r0, kernel_type, alpha, method, interval_data));
    return rcpp_result_gen;
END_RCPP
}
// test_const_column
bool test_const_column(const MapMat& d);
RcppExport SEXP _RobustGaSP_test_const_column(SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const MapMat& >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(test_const_column(d));
    return rcpp_result_gen;
END_RCPP
}
