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
// log_marginal_lik
double log_marginal_lik(const Vec param, double nugget, const bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const String kernel_type, const Eigen::VectorXd alpha);
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
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
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
Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const String kernel_type, const Eigen::VectorXd alpha);
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
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_lik_deriv(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
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
double log_ref_marginal_post(const Eigen::VectorXd param, double nugget, bool nugget_est, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const String kernel_type, const Eigen::VectorXd alpha);
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
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_ref_marginal_post(param, nugget, nugget_est, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// construct_rgasp
List construct_rgasp(const Eigen::VectorXd beta, const double nu, const List R0, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const String kernel_type, const Eigen::VectorXd alpha);
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
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(construct_rgasp(beta, nu, R0, X, zero_mean, output, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// pred_rgasp
List pred_rgasp(const Eigen::VectorXd beta, const double nu, const Eigen::Map<Eigen::MatrixXd>& input, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const Eigen::Map<Eigen::MatrixXd>& testing_input, const Eigen::Map<Eigen::MatrixXd>& X_testing, const Eigen::Map<Eigen::MatrixXd>& L, Eigen::Map<Eigen::MatrixXd>& LX, Eigen::Map<Eigen::VectorXd>& theta_hat, double sigma2_hat, double qt_025, double qt_975, List r0, const String kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_pred_rgasp(SEXP betaSEXP, SEXP nuSEXP, SEXP inputSEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP testing_inputSEXP, SEXP X_testingSEXP, SEXP LSEXP, SEXP LXSEXP, SEXP theta_hatSEXP, SEXP sigma2_hatSEXP, SEXP qt_025SEXP, SEXP qt_975SEXP, SEXP r0SEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
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
    Rcpp::traits::input_parameter< double >::type qt_025(qt_025SEXP);
    Rcpp::traits::input_parameter< double >::type qt_975(qt_975SEXP);
    Rcpp::traits::input_parameter< List >::type r0(r0SEXP);
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(pred_rgasp(beta, nu, input, X, zero_mean, output, testing_input, X_testing, L, LX, theta_hat, sigma2_hat, qt_025, qt_975, r0, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
// generate_predictive_mean_cov
List generate_predictive_mean_cov(const Eigen::VectorXd beta, const double nu, const Eigen::Map<Eigen::MatrixXd>& input, const Eigen::Map<Eigen::MatrixXd>& X, const String zero_mean, const Eigen::Map<Eigen::MatrixXd>& output, const Eigen::Map<Eigen::MatrixXd>& testing_input, const Eigen::Map<Eigen::MatrixXd>& X_testing, const Eigen::Map<Eigen::MatrixXd>& L, Eigen::Map<Eigen::MatrixXd>& LX, Eigen::Map<Eigen::VectorXd>& theta_hat, double sigma2_hat, List rr0, List r0, const String kernel_type, const Eigen::VectorXd alpha);
RcppExport SEXP _RobustGaSP_generate_predictive_mean_cov(SEXP betaSEXP, SEXP nuSEXP, SEXP inputSEXP, SEXP XSEXP, SEXP zero_meanSEXP, SEXP outputSEXP, SEXP testing_inputSEXP, SEXP X_testingSEXP, SEXP LSEXP, SEXP LXSEXP, SEXP theta_hatSEXP, SEXP sigma2_hatSEXP, SEXP rr0SEXP, SEXP r0SEXP, SEXP kernel_typeSEXP, SEXP alphaSEXP) {
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
    Rcpp::traits::input_parameter< const String >::type kernel_type(kernel_typeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(generate_predictive_mean_cov(beta, nu, input, X, zero_mean, output, testing_input, X_testing, L, LX, theta_hat, sigma2_hat, rr0, r0, kernel_type, alpha));
    return rcpp_result_gen;
END_RCPP
}
