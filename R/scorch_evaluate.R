#===============================================================================
# PREDICTION AND EVALUATION HELPERS
#===============================================================================

#' Predict with a Scorch Model
#'
#' @param scorch_model A \code{scorch_model} object.
#' @param input A torch tensor or named list of torch tensors.
#'
#' @returns Model predictions.
#'
#' @family model training
#'
#' @export
scorch_predict <- function(scorch_model, input) {
  scorch_model <- scorch_check_model(scorch_model)

  model <- if (isTRUE(scorch_model$compiled) &&
               !is.null(scorch_model$nn_model)) {
    scorch_model$nn_model
  } else {
    as_torch_module(scorch_model)
  }

  model$eval()

  #- Detect the device the model lives on from its first parameter, then move
  #- all input tensors to that device so predict works after GPU/MPS training.
  params <- model$parameters
  model_device <- if (length(params) > 0) params[[1]]$device else NULL

  input <- scorch_as_named_tensor_list(input, default_name = "input")

  if (!is.null(model_device)) {
    input <- lapply(input, function(x) {
      if (inherits(x, "torch_tensor")) x$to(device = model_device) else x
    })
  }

  torch::with_no_grad({
    if (length(scorch_model$inputs) == 1) {
      model(input[[1]])
    } else {
      do.call(model, input[scorch_model$inputs])
    }
  })
}

#' Evaluate Predictions
#'
#' @param predictions A torch tensor of model predictions.
#' @param truth A torch tensor of observed outcomes.
#' @param metric Evaluation metric. Currently \code{"accuracy"} and
#'   \code{"mse"} are supported.
#'
#' @returns A one-row tibble with the metric value.
#'
#' @family model training
#'
#' @export
scorch_evaluate_predictions <- function(predictions,
                                        truth,
                                        metric = c("accuracy", "mse")) {
  metric <- match.arg(metric)

  if (!inherits(predictions, "torch_tensor") ||
      !inherits(truth, "torch_tensor")) {
    stop("`predictions` and `truth` must be torch tensors.", call. = FALSE)
  }

  truth <- truth$to(device = predictions$device)

  value <- switch(
    metric,
    accuracy = {
      pred_class <- if (predictions$dim() > 1L) {
        torch::torch_argmax(predictions, dim = 2)
      } else {
        predictions
      }
      correct <- (pred_class == truth)$to(dtype = torch::torch_float())
      torch::torch_mean(correct)$item()
    },
    mse = {
      torch::nnf_mse_loss(predictions, truth)$item()
    }
  )

  tibble::tibble(metric = metric, value = value)
}

#' Evaluate a Scorch Model
#'
#' @param scorch_model A \code{scorch_model} object.
#' @param input A torch tensor or named list of torch tensors.
#' @param truth A torch tensor of observed outcomes.
#' @param metric Evaluation metric. Currently \code{"accuracy"} and
#'   \code{"mse"} are supported.
#'
#' @returns A one-row tibble with the metric value.
#'
#' @family model training
#'
#' @export
scorch_evaluate <- function(scorch_model,
                            input,
                            truth,
                            metric = c("accuracy", "mse")) {
  predictions <- scorch_predict(scorch_model, input)
  scorch_evaluate_predictions(predictions, truth, metric = metric)
}

#=== END =======================================================================
